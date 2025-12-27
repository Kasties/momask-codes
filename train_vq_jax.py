"""
JAX Training Script for VQ-VAE
Converted from train_vq.py for TPU training

Usage:
    python train_vq_jax.py --dataset_name t2m --batch_size 256 --name rvq_jax_test
"""

import os
import sys
import time
import argparse
from os.path import join as pjoin
from collections import defaultdict, OrderedDict

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

# Add parent directory to path for imports
try:
    # Normal execution
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_script_dir))
    sys.path.insert(0, _script_dir)
except NameError:
    # exec() in notebook - assume we're in the right directory or files are in path
    pass

from models.vq.model_jax import RVQVAE


def arg_parse(is_train=False):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataloader
    parser.add_argument('--dataset_name', type=str, default='t2m', help='dataset name: t2m or kit')
    parser.add_argument('--data_root', type=str, default=None, help='custom path to dataset root')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')

    # Optimization
    parser.add_argument('--max_epoch', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of warmup iterations')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="commitment loss weight")
    parser.add_argument('--loss_vel', type=float, default=0.5, help='velocity loss weight')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')

    # VQ-VAE architecture
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=512, help="number of embeddings")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="num of resblocks")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq_act', type=str, default='relu', help='activation function')
    parser.add_argument('--vq_norm', type=str, default=None, help='normalization')

    parser.add_argument('--num_quantizers', type=int, default=6, help='number of quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize dropout prob')

    # Other
    parser.add_argument('--name', type=str, default="test_jax", help='Name of this trial')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='checkpoints directory')
    parser.add_argument('--log_every', default=10, type=int, help='logging frequency')
    parser.add_argument('--save_every', default=1000, type=int, help='save frequency (iterations)')
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument('--is_continue', action="store_true", help='continue training')

    opt = parser.parse_args()
    opt.is_train = is_train
    return opt


def load_motion_data(data_root, split_file, window_size=64):
    """Load motion data from numpy files.
    
    Args:
        data_root: Path to dataset root
        split_file: Path to split file (train.txt or val.txt)
        window_size: Window size for motion clips
        
    Returns:
        data: numpy array of shape (N, window_size, dim_pose)
    """
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    
    data = []
    with open(split_file, 'r') as f:
        for line in f.readlines():
            name = line.strip()
            motion_path = pjoin(data_root, 'new_joint_vecs', name + '.npy')
            if os.path.exists(motion_path):
                motion = np.load(motion_path)
                # Normalize
                motion = (motion - mean) / std
                
                # Extract windows
                if len(motion) >= window_size:
                    # Take random windows during training
                    for i in range(0, len(motion) - window_size + 1, window_size // 2):
                        window = motion[i:i+window_size]
                        data.append(window)
    
    return np.array(data, dtype=np.float32), mean, std


def compute_loss(model, batch, commit_weight=0.02, vel_weight=0.5, joints_num=22):
    """Compute reconstruction and commitment losses."""
    pred, commit_loss, perplexity = model(batch)
    
    # Reconstruction loss (L1)
    loss_rec = jnp.mean(jnp.abs(pred - batch))
    
    # Explicit position loss (for joints) 
    pred_local_pos = pred[..., 4:(joints_num - 1) * 3 + 4]
    local_pos = batch[..., 4:(joints_num - 1) * 3 + 4]
    loss_vel = jnp.mean(jnp.abs(pred_local_pos - local_pos))
    
    # Total loss
    total_loss = loss_rec + vel_weight * loss_vel + commit_weight * commit_loss
    
    return total_loss, {
        'loss': total_loss,
        'loss_rec': loss_rec,
        'loss_vel': loss_vel,
        'loss_commit': commit_loss,
        'perplexity': perplexity,
    }


def create_train_step(optimizer, commit_weight, vel_weight, joints_num):
    """Create JIT-compiled training step."""
    
    def train_step(model, opt_state, batch):
        def loss_fn(m):
            loss, aux = compute_loss(m, batch, commit_weight, vel_weight, joints_num)
            return loss, aux
        
        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        
        # Update using nnx.Optimizer pattern
        return loss, aux, grads
    
    return train_step


def main():
    opt = arg_parse(is_train=True)
    
    # Set up random seed
    np.random.seed(opt.seed)
    key = jax.random.PRNGKey(opt.seed)
    
    # Print JAX devices
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    
    # Dataset configuration
    if opt.dataset_name == "t2m":
        default_data_root = './dataset/HumanML3D/'
        opt.joints_num = 22
        dim_pose = 263
    elif opt.dataset_name == "kit":
        default_data_root = './dataset/KIT-ML/'
        opt.joints_num = 21
        dim_pose = 251
    else:
        raise KeyError(f'Dataset {opt.dataset_name} does not exist')
    
    # Use custom data_root if provided
    opt.data_root = opt.data_root if opt.data_root else default_data_root
    
    # Create directories
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    os.makedirs(opt.model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training VQ-VAE with JAX")
    print(f"Dataset: {opt.dataset_name}")
    print(f"Batch size: {opt.batch_size}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading training data...")
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')
    
    if not os.path.exists(train_split_file):
        print(f"ERROR: Training split file not found at {train_split_file}")
        print("Please ensure the dataset is properly set up.")
        print("\nFor testing without dataset, creating dummy data...")
        
        # Create dummy data for testing
        n_train = 1000
        n_val = 100
        train_data = np.random.randn(n_train, opt.window_size, dim_pose).astype(np.float32)
        val_data = np.random.randn(n_val, opt.window_size, dim_pose).astype(np.float32)
        mean, std = np.zeros(dim_pose), np.ones(dim_pose)
    else:
        train_data, mean, std = load_motion_data(opt.data_root, train_split_file, opt.window_size)
        val_data, _, _ = load_motion_data(opt.data_root, val_split_file, opt.window_size)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create model
    print("\nInitializing model...")
    rngs = nnx.Rngs(opt.seed)
    
    model = RVQVAE(
        args=opt,
        input_width=dim_pose,
        nb_code=opt.nb_code,
        code_dim=opt.code_dim,
        output_emb_width=opt.code_dim,
        down_t=opt.down_t,
        stride_t=opt.stride_t,
        width=opt.width,
        depth=opt.depth,
        dilation_growth_rate=opt.dilation_growth_rate,
        activation=opt.vq_act,
        norm=opt.vq_norm,
        rngs=rngs
    )
    
    # Count parameters properly for NNX
    def count_params(state):
        total = 0
        for leaf in jax.tree_util.tree_leaves(state):
            if hasattr(leaf, 'value'):
                arr = leaf.value
                if hasattr(arr, 'size'):
                    total += arr.size
            elif isinstance(leaf, jnp.ndarray):
                total += leaf.size
        return total
    
    graphdef, state = nnx.split(model)
    n_params = count_params(state)
    print(f"Total parameters: {n_params / 1e6:.2f}M")
    
    # Create optimizer
    n_batches = max(1, len(train_data) // opt.batch_size)
    total_steps = opt.max_epoch * n_batches
    warmup_steps = min(opt.warm_up_iter, total_steps // 4)  # Cap warmup at 25% of total
    
    print(f"LR schedule: warmup={warmup_steps}, total={total_steps}")
    
    # Use join_schedules for more control
    if warmup_steps > 0 and total_steps > warmup_steps:
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=0.0, end_value=opt.lr, transition_steps=warmup_steps),
                optax.cosine_decay_schedule(init_value=opt.lr, decay_steps=total_steps - warmup_steps, alpha=0.01),
            ],
            boundaries=[warmup_steps]
        )
    else:
        # Simple constant LR for very small datasets
        lr_schedule = opt.lr
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=opt.weight_decay)
    )
    
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    
    # Training loop
    print("\nStarting training...")
    total_iters = total_steps
    
    epoch = 0
    it = 0
    start_time = time.time()
    logs = defaultdict(float)
    
    while epoch < opt.max_epoch:
        # Shuffle data
        key, subkey = jax.random.split(key)
        perm = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[perm]
        
        for i in range(n_batches):
            it += 1
            
            # Get batch
            batch = jnp.array(train_data_shuffled[i * opt.batch_size:(i + 1) * opt.batch_size])
            
            # Compute loss and gradients
            def loss_fn(m):
                loss, aux = compute_loss(m, batch, opt.commit, opt.loss_vel, opt.joints_num)
                return loss, aux
            
            (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(grads)
            
            # Log
            for k, v in aux.items():
                logs[k] += float(v)
            
            # Print logs
            if it % opt.log_every == 0:
                elapsed = time.time() - start_time
                mean_logs = {k: v / opt.log_every for k, v in logs.items()}
                
                log_str = f"[{elapsed:.0f}s] Epoch {epoch} Iter {it}/{total_iters}"
                log_str += f" | loss: {mean_logs['loss']:.4f}"
                log_str += f" | rec: {mean_logs['loss_rec']:.4f}"
                log_str += f" | commit: {mean_logs['loss_commit']:.4f}"
                log_str += f" | perp: {mean_logs['perplexity']:.1f}"
                print(log_str)
                
                logs = defaultdict(float)
            
            # Save checkpoint
            if it % opt.save_every == 0:
                save_path = pjoin(opt.model_dir, 'latest.pkl')
                import pickle
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'model_state': nnx.state(model),
                        'opt_state': optimizer.opt_state,
                        'epoch': epoch,
                        'iter': it,
                    }, f)
                print(f"Saved checkpoint to {save_path}")
        
        # End of epoch
        epoch += 1
        
        # Validation
        print(f"\n--- Epoch {epoch} Validation ---")
        val_logs = defaultdict(list)
        n_val_batches = len(val_data) // opt.batch_size
        
        for i in range(n_val_batches):
            batch = jnp.array(val_data[i * opt.batch_size:(i + 1) * opt.batch_size])
            _, aux = compute_loss(model, batch, opt.commit, opt.loss_vel, opt.joints_num)
            for k, v in aux.items():
                val_logs[k].append(float(v))
        
        val_summary = {k: np.mean(v) for k, v in val_logs.items()}
        print(f"Val loss: {val_summary['loss']:.4f} | rec: {val_summary['loss_rec']:.4f} | perp: {val_summary['perplexity']:.1f}")
        print()
        
        # Save epoch checkpoint
        save_path = pjoin(opt.model_dir, f'epoch_{epoch:03d}.pkl')
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model_state': nnx.state(model),
                'epoch': epoch,
            }, f)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
