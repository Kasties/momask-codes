import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optax # Common for losses, though we can use jnp

def gumbel_sample(logits, key, temperature=1.0, stochastic=False, training=True):
    if training and stochastic and temperature > 0:
        # JAX random generation
        u = jax.random.uniform(key, logits.shape)
        noise = -jnp.log(-jnp.log(u + 1e-20) + 1e-20)
        sampling_logits = (logits / temperature) + noise
    else:
        sampling_logits = logits

    return jnp.argmax(sampling_logits, axis=-1)

class QuantizeEMAReset(nnx.Module):
    def __init__(self, nb_code, code_dim, args, rngs: nnx.Rngs):
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.get('mu', 0.99)
        self.rngs = rngs
        
        # Initialize state
        self.init = False
        # Use nnx.Variable for state that isn't a gradient-updated parameter
        self.codebook = nnx.Variable(jnp.zeros((nb_code, code_dim)))
        self.code_sum = nnx.Variable(jnp.zeros((nb_code, code_dim)))
        self.code_count = nnx.Variable(jnp.zeros(nb_code))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / jnp.sqrt(code_dim)
            out = jnp.tile(x, (n_repeats, 1))
            # Get a new key from the rng stream
            noise = jax.random.normal(self.rngs.params(), out.shape) * std
            out = out + noise
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook.value = out[:self.nb_code]
        self.code_sum.value = self.codebook.value
        self.code_count.value = jnp.ones(self.nb_code)
        self.init = True

    def quantize(self, x, training=True, temperature=0.):
        # x: [NT, C], codebook: [N, C]
        # Distance calculation: (a-b)^2 = a^2 - 2ab + b^2
        distances = (jnp.sum(x**2, axis=-1, keepdims=True) 
                    - 2 * jnp.dot(x, self.codebook.value.T) 
                    + jnp.sum(self.codebook.value**2, axis=-1))

        # Use the params RNG stream for gumbel noise
        key = self.rngs.params() 
        code_idx = gumbel_sample(-distances, key, temperature=temperature, 
                                stochastic=True, training=training)
        return code_idx

    def dequantize(self, indices):
        return jnp.take(self.codebook.value, indices, axis=0)

    def compute_perplexity(self, code_idx):
        code_onehot = jax.nn.one_hot(code_idx, self.nb_code)
        code_count = jnp.sum(code_onehot, axis=0)
        prob = code_count / (jnp.sum(code_count) + 1e-10)
        perplexity = jnp.exp(-jnp.sum(prob * jnp.log(prob + 1e-7)))
        return perplexity

    def update_codebook(self, x, code_idx):
        # x: [NT, C], code_idx: [NT]
        code_onehot = jax.nn.one_hot(code_idx, self.nb_code) # [NT, nb_code]
        
        # New counts and sums
        new_code_sum = jnp.dot(code_onehot.T, x) # [nb_code, C]
        new_code_count = jnp.sum(code_onehot, axis=0) # [nb_code]

        # EMA Update
        self.code_sum.value = self.mu * self.code_sum.value + (1. - self.mu) * new_code_sum
        self.code_count.value = self.mu * self.code_count.value + (1. - self.mu) * new_code_count

        # Cluster reset for unused codes
        usage = (self.code_count.value >= 1.0)[:, None]
        code_update = self.code_sum.value / jnp.maximum(self.code_count.value[:, None], 1e-10)
        
        # Tile a random batch of x to replace dead codes
        out_random = self._tile(x)[:self.nb_code]
        self.codebook.value = usage * code_update + (1 - usage) * out_random

        return self.compute_perplexity(code_idx)

    def __call__(self, x, training=True, return_idx=False, temperature=0.):
        N, C, T = x.shape
        
        # Preprocess: NCT -> (NT) C
        x_flat = rearrange(x, 'n c t -> (n t) c')
        
        if training and not self.init:
            self.init_codebook(x_flat)

        code_idx = self.quantize(x_flat, training=training, temperature=temperature)
        x_d = self.dequantize(code_idx)

        if training:
            perplexity = self.update_codebook(x_flat, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Commitment Loss
        commit_loss = jnp.mean((x_flat - jax.lax.stop_gradient(x_d))**2)

        # Straight-through estimator
        x_d_st = x_flat + jax.lax.stop_gradient(x_d - x_flat)

        # Postprocess: (NT) C -> N C T
        x_d_out = rearrange(x_d_st, '(n t) c -> n c t', n=N, t=T)
        code_idx_out = rearrange(code_idx, '(n t) -> n t', n=N, t=T)

        if return_idx:
            return x_d_out, code_idx_out, commit_loss, perplexity
        return x_d_out, commit_loss, perplexity

# --- Test Script ---
#args = {'mu': 0.99}
#rngs = nnx.Rngs(0)
#quantizer = QuantizeEMAReset(nb_code=512, code_dim=64, args=args, rngs=rngs)
#
#x = jnp.ones((4, 64, 32))  # (batch, dim, time)
## In NNX, we usually pass training as a boolean
#x_q, idx, loss, perp = quantizer(x, training=True, return_idx=True)
#
#print(f"Output shape: {x_q.shape}") # (4, 64, 32)
#print(f"Indices shape: {idx.shape}") # (4, 32)
#print(f"Perplexity: {perp}")