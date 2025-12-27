import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

#from models.vq.resnet_jax import Resnet1D
from resnet_jax import Resnet1D


def get_activation(name):
    if name == 'relu': return jax.nn.relu
    if name == 'leaky_relu': return jax.nn.leaky_relu
    if name == 'gelu': return jax.nn.gelu
    return jax.nn.relu


def get_norm(name, dim, rngs):
    if name == 'layer':
        return nnx.LayerNorm(dim, rngs=rngs)
    # Add others (batch, etc) if needed, otherwise return identity
    return lambda x: x
# --- 2. Encoder ---
class Encoder(nnx.Module):
    def __init__(self, input_emb_width=263, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, rngs=nnx.Rngs,activation='relu', norm='layer'):
        blocks = []
        # Initial projection
        blocks.append(nnx.Conv(input_emb_width, width, kernel_size=(3,), padding=1, rngs=rngs))
        blocks.append(get_activation(activation))
        blocks.append(get_norm(norm, width, rngs))

        filter_t = stride_t * 2
        pad_t = stride_t // 2

        for i in range(down_t):
            blocks.append(nnx.Conv(width, width, kernel_size=(filter_t,), strides=stride_t, padding=pad_t, rngs=rngs))
            blocks.append(Resnet1D(width, depth, dilation_growth_rate, rngs=rngs))
        
        blocks.append(nnx.Conv(width, output_emb_width, kernel_size=(3,), padding=1, rngs=rngs))
        self.model = nnx.Sequential(*blocks)

    def __call__(self, x):
        # Convert (B, C, T) -> (B, T, C) for Flax Conv
        x = rearrange(x, 'b c t -> b t c')
        x = self.model(x)
        return rearrange(x, 'b t c -> b c t')

# --- 3. Decoder ---
class Decoder(nnx.Module):
    def __init__(self, input_emb_width=263, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, rngs=nnx.Rngs,activation='relu', norm='layer'):
        blocks = []
        blocks.append(nnx.Conv(output_emb_width, width, kernel_size=(3,), padding=1, rngs=rngs))
        blocks.append(get_activation(activation))
        blocks.append(get_norm(norm, width, rngs))

        for i in range(down_t):
            blocks.append(Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, rngs=rngs))
            # Upsampling
            blocks.append(lambda x: jax.image.resize(x, (x.shape[0], x.shape[1] * stride_t, x.shape[2]), method='nearest'))
            blocks.append(nnx.Conv(width, width, kernel_size=(3,), padding=1, rngs=rngs))

        blocks.append(get_activation(activation))
        blocks.append(nnx.Conv(width, input_emb_width, kernel_size=(3,), padding=1, rngs=rngs))
        self.model = nnx.Sequential(*blocks)

    def __call__(self, x):
        x = rearrange(x, 'b c t -> b t c')
        x = self.model(x)
        return rearrange(x, 'b t c -> b c t')

# --- 4. Test Script ---
#rngs = nnx.Rngs(0)
#enc = Encoder(input_emb_width=263, output_emb_width=512, rngs=rngs)
#dec = Decoder(input_emb_width=263, output_emb_width=512, rngs=rngs)
#
#x = jnp.ones((4, 263, 64))  # (batch, features, time)
#z = enc(x)
#print(f"Encoded shape: {z.shape}")  # (4, 512, 16)
#
#x_recon = dec(z)
#print(f"Decoded shape: {x_recon.shape}")  # (4, 263, 64)