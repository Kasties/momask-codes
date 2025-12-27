import random

import jax
import jax.numpy as jnp
from flax import nnx
try:
    from models.vq.encdec_jax import Encoder, Decoder
    from models.vq.residual_vq_jax import ResidualVQ
except ImportError:
    from encdec_jax import Encoder, Decoder
    from residual_vq_jax import ResidualVQ

class RVQVAE(nnx.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 rngs=nnx.Rngs):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, rngs=rngs)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, rngs=rngs)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim': code_dim, 
        }
        self.quantizer = ResidualVQ(**rvqvae_config, rngs=rngs)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.transpose(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.transpose(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def __call__(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        # print(code_idx[0, :, 1])
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(axis=0).transpose(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

class LengthEstimator(nnx.Module):
    def __init__(self, input_size, output_size, rngs=nnx.Rngs):
        super().__init__()
        nd = 512
        # Initialize weights with normal(0.02)
        k_init = nnx.initializers.normal(stddev=0.02)
        b_init = nnx.initializers.zeros
        
        self.output = nnx.Sequential(
            nnx.Linear(input_size, nd, kernel_init=k_init, bias_init=b_init, rngs=rngs),
            nnx.LayerNorm(nd, rngs=rngs),
            lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            nnx.Dropout(0.2, rngs=rngs),
            
            nnx.Linear(nd, nd // 2, kernel_init=k_init, bias_init=b_init, rngs=rngs),
            nnx.LayerNorm(nd // 2, rngs=rngs),
            lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            nnx.Dropout(0.2, rngs=rngs),
            
            nnx.Linear(nd // 2, nd // 4, kernel_init=k_init, bias_init=b_init, rngs=rngs),
            nnx.LayerNorm(nd // 4, rngs=rngs),
            lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            
            nnx.Linear(nd // 4, output_size, kernel_init=k_init, bias_init=b_init, rngs=rngs)
        )

    def __call__(self, text_emb):
        return self.output(text_emb)

if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np
    from types import SimpleNamespace

    # Create args namespace with required parameters
    args = SimpleNamespace(
        num_quantizers=6,
        shared_codebook=True,
        quantize_dropout_prob=0.2,
    )

    # Create RNG key for model initialization
    rngs = nnx.Rngs(0)

    # Load same random input
    np_input = np.random.randn(4, 64, 263).astype(np.float32)

    # JAX forward  
    jax_model = RVQVAE(
        args=args,
        input_width=263,
        nb_code=1024,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        rngs=rngs
    )
    jax_out, commit_loss, perplexity = jax_model(jnp.array(np_input))

    print(f"Input shape: {np_input.shape}")
    print(f"Output shape: {jax_out.shape}")
    print(f"Commit loss: {commit_loss}")
    print(f"Perplexity: {perplexity}")
    print("✓ Full model forward pass successful!")