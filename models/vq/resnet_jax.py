import jax
import jax.numpy as jnp
from flax import nnx

class nonlinearity(nnx.Module):
    def __call__(self, x):
        return x * nnx.sigmoid(x)

class ResConv1DBlock(nnx.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2, rngs=None):
        super().__init__()

        self.norm_type = norm

        # Norm layers in Flax operate on the last dimension by default
        if norm == "LN":
            self.norm1 = nnx.LayerNorm(num_features=n_in, rngs=rngs)
            self.norm2 = nnx.LayerNorm(num_features=n_in, rngs=rngs)
        elif norm == "GN":
            self.norm1 = nnx.GroupNorm(num_groups=32, num_features=n_in, rngs=rngs)
            self.norm2 = nnx.GroupNorm(num_groups=32, num_features=n_in, rngs=rngs)
        elif norm == "BN":
            self.norm1 = nnx.BatchNorm1d(num_features=n_in, rngs=rngs)
            self.norm2 = nnx.BatchNorm1d(num_features=n_in, rngs=rngs)
        else:
            self.norm1 = jax.nn.identity
            self.norm2 = jax.nn.identity

        if activation == "relu":
            self.activation1 = jax.nn.relu
            self.activation2 = jax.nn.relu
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = jax.nn.gelu
            self.activation2 = jax.nn.gelu

        # Conv expects (Batch, Time, Channels)
        # padding=dilation with kernel_size=3 maintains sequence length
        self.conv1 = nnx.Conv(
            in_features=n_in,
            out_features=n_state,
            kernel_size=(3,),
            kernel_dilation=(dilation,),
            padding=dilation, 
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=n_state,
            out_features=n_in,
            kernel_size=(1,),
            padding=0,
            rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x):
        x_orig = x
        
        # 1st Block: Norm -> Act -> Conv
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)

        # 2nd Block: Norm -> Act -> Conv
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        
        x = self.dropout(x)
        return x + x_orig

class Resnet1D(nnx.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None, rngs=None):
        super().__init__()

        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, 
                           activation=activation, norm=norm, rngs=rngs)
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nnx.Sequential(*blocks)

    def __call__(self, x):
        return self.model(x)

# --- Test ---
# Use (Batch, Time, Channels) format
#batch, channels, time = 4, 512, 64
#x = jnp.ones((batch, time, channels)) 
#
#model = Resnet1D(n_in=channels, n_depth=3, rngs=nnx.Rngs(0))
#y = model(x)
#
#assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
#print(f"✓ Resnet1D: shapes match. Output shape: {y.shape}")