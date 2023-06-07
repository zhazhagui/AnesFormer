import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='OBA', help="The problem to solve, default 'OBA'")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    

    # Model
    parser.add_argument('--model', default='attention', help="'attention' (default)")
    parser.add_argument('--input_dim', type=int, default=40, help='Dimension of input embedding')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--output_dim', type=int, default=2, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='layer', help="Normalization type, 'batch' (default) or 'instance'")

    # Trainin
    parser.add_argument('--lr_meta_model', type=float, default=1e-4, help="Set the learning rate for the DGF network")
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the attention network")
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=15, help='The number of epochs to train')
    parser.add_argument('--query_inner_loop', type=int, default=1, help='The number of query_inner_loop to train')
    parser.add_argument('--support_inner_loop', type=int, default=1, help='The number of support_inner_loop to train')
    parser.add_argument('--task_num', type=int, default=1, help='The number of train_task')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=0.95,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout'")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
   
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help="Batch size to use during (baseline) evaluation")
    

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', default=None,help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--alpha', type=float, default=1.0, help='Hyperparameter for training. The parameter for VAE')
     
    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        opts.run_name
    )

    assert (opts.baseline == 'rollout')
    return opts
