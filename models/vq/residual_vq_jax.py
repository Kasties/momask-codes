import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange, repeat
import random

# --- Helper: Updated QuantizeEMAReset from previous step ---

class QuantizeEMAReset(nnx.Module):
    def __init__(self, nb_code, code_dim, mu=0.99, rngs=None):
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.rngs = rngs
        
        self.init = False
        self.codebook = nnx.Variable(jnp.zeros((nb_code, code_dim)))
        self.code_sum = nnx.Variable(jnp.zeros((nb_code, code_dim)))
        self.code_count = nnx.Variable(jnp.zeros(nb_code))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            out = jnp.tile(x, (n_repeats, 1))
            noise = jax.random.normal(self.rngs.params(), out.shape) * (0.01 / jnp.sqrt(code_dim))
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
        distances = (jnp.sum(x**2, axis=-1, keepdims=True) 
                    - 2 * jnp.dot(x, self.codebook.value.T) 
                    + jnp.sum(self.codebook.value**2, axis=-1))
        
        if training and temperature > 0:
            u = jax.random.uniform(self.rngs.params(), distances.shape)
            noise = -jnp.log(-jnp.log(u + 1e-20) + 1e-20)
            logits = (-distances / temperature) + noise
        else:
            logits = -distances
        return jnp.argmax(logits, axis=-1)

    def dequantize(self, indices):
        return jnp.take(self.codebook.value, indices, axis=0)

    def update_codebook(self, x, code_idx):
        code_onehot = jax.nn.one_hot(code_idx, self.nb_code)
        new_code_sum = jnp.dot(code_onehot.T, x)
        new_code_count = jnp.sum(code_onehot, axis=0)

        self.code_sum.value = self.mu * self.code_sum.value + (1. - self.mu) * new_code_sum
        self.code_count.value = self.mu * self.code_count.value + (1. - self.mu) * new_code_count

        usage = (self.code_count.value >= 1.0)[:, None]
        code_update = self.code_sum.value / jnp.maximum(self.code_count.value[:, None], 1e-10)
        out_random = self._tile(x)[:self.nb_code]
        self.codebook.value = usage * code_update + (1 - usage) * out_random

        prob = new_code_count / (jnp.sum(new_code_count) + 1e-10)
        return jnp.exp(-jnp.sum(prob * jnp.log(prob + 1e-7)))

    def __call__(self, x, training=True, return_idx=False, temperature=0.):
        N, C, T = x.shape
        x_flat = rearrange(x, 'n c t -> (n t) c')
        
        if training and not self.init:
            self.init_codebook(x_flat)

        code_idx = self.quantize(x_flat, training=training, temperature=0.)
        x_d = self.dequantize(code_idx)

        if training:
            perplexity = self.update_codebook(x_flat, code_idx)
        else:
            code_onehot = jax.nn.one_hot(code_idx, self.nb_code)
            prob = jnp.sum(code_onehot, axis=0) / (x_flat.shape[0] + 1e-10)
            perplexity = jnp.exp(-jnp.sum(prob * jnp.log(prob + 1e-7)))

        commit_loss = jnp.mean((x_flat - jax.lax.stop_gradient(x_d))**2)
        x_d_st = x_flat + jax.lax.stop_gradient(x_d - x_flat)

        x_d_out = rearrange(x_d_st, '(n t) c -> n c t', n=N, t=T)
        code_idx_out = rearrange(code_idx, '(n t) -> n t', n=N, t=T)

        return (x_d_out, code_idx_out, commit_loss, perplexity) if return_idx else (x_d_out, commit_loss, perplexity)

# --- Main ResidualVQ Class ---

class ResidualVQ(nnx.Module):
    def __init__(
        self,
        num_quantizers,
        nb_code,
        code_dim,
        mu=0.99,
        shared_codebook=False,
        quantize_dropout_prob=0.1,
        quantize_dropout_cutoff_index=0,
        rngs: nnx.Rngs = None,
    ):
        self.num_quantizers = num_quantizers
        self.rngs = rngs
        self.quantize_dropout_prob = quantize_dropout_prob
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index

        if shared_codebook:
            layer = QuantizeEMAReset(nb_code, code_dim, mu, rngs=rngs)
            self.layers = [layer for _ in range(num_quantizers)]
        else:
            self.layers = [QuantizeEMAReset(nb_code, code_dim, mu, rngs=rngs) for _ in range(num_quantizers)]

    @property
    def codebooks(self):
        # Extract codebook values from all layers
        return jnp.stack([l.codebook.value for l in self.layers], axis=0)

    def get_codes_from_indices(self, indices):
        # indices shape: [batch, time, q]
        # JAX version of gather
        b, t, q = indices.shape
        all_codes = []
        for i in range(q):
            layer_indices = indices[:, :, i]
            # Handle dropout indices (-1)
            mask = (layer_indices != -1)
            clean_indices = jnp.where(mask, layer_indices, 0)
            codes = self.layers[i].dequantize(clean_indices)
            codes = jnp.where(mask[..., None], codes, 0.0)
            all_codes.append(codes)
        
        return jnp.stack(all_codes, axis=0) # [q, b, t, d]

    def __call__(self, x, training=True, sample_codebook_temp=0., force_dropout_index=-1):
        N, C, T = x.shape
        quantized_out = jnp.zeros_like(x)
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []

        # Dropout logic using JAX random if possible, or python random for structural
        should_dropout = training and (random.random() < self.quantize_dropout_prob)
        start_drop_idx = self.num_quantizers
        
        if force_dropout_index >= 0:
            should_dropout = True
            start_drop_idx = force_dropout_index
        elif should_dropout:
            start_drop_idx = random.randrange(self.quantize_dropout_cutoff_index, self.num_quantizers)

        for i, layer in enumerate(self.layers):
            if should_dropout and i > start_drop_idx:
                all_indices.append(jnp.full((N, T), -1, dtype=jnp.int32))
                continue

            # Quantize
            quantized, indices, loss, perp = layer(residual, training=training, return_idx=True, temperature=sample_codebook_temp)
            
            # Update residual (stop gradient on the quantized value for the next layer's input)
            residual = residual - jax.lax.stop_gradient(quantized)
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_perplexity.append(perp)

        all_indices = jnp.stack(all_indices, axis=-1)
        avg_loss = jnp.mean(jnp.array(all_losses))
        avg_perp = jnp.mean(jnp.array(all_perplexity))

        return quantized_out, all_indices, avg_loss, avg_perp

    def quantize(self, x, return_latent=False):
        """Encode-only quantization without training updates.
        
        Args:
            x: Input tensor of shape (N, C, T)
            return_latent: If True, also return all quantized codes
            
        Returns:
            code_idx: Indices of shape (N, T, num_quantizers)
            all_codes: (optional) Quantized codes of shape (num_quantizers, N, C, T)
        """
        all_indices = []
        quantized_out = jnp.zeros_like(x)
        residual = x
        all_codes = []
        
        for quantizer_index, layer in enumerate(self.layers):
            # Quantize without training updates
            quantized, indices, loss, perplexity = layer(residual, training=False, return_idx=True)
            
            residual = residual - jax.lax.stop_gradient(quantized)
            quantized_out = quantized_out + quantized
            
            all_indices.append(indices)
            all_codes.append(quantized)
        
        code_idx = jnp.stack(all_indices, axis=-1)
        all_codes = jnp.stack(all_codes, axis=0)
        
        if return_latent:
            return code_idx, all_codes
        return code_idx

# --- Test ---
#rngs = nnx.Rngs(0)
#rvq = ResidualVQ(num_quantizers=6, nb_code=512, code_dim=64, rngs=rngs)
#x = jnp.ones((4, 64, 32))
#
## Run forward pass
#x_q, indices, loss, perp = rvq(x, training=True)
#
#print(f"✓ Output shape: {x_q.shape}")     # (4, 64, 32)
#print(f"✓ Indices shape: {indices.shape}") # (4, 32, 6)
#print(f"✓ Loss: {loss:.4f}, Perp: {perp:.4f}")
#assert indices.shape == (4, 32, 6)  # (batch, time, num_quantizers)
#print("✓ ResidualVQ: multi-layer quantization works")
