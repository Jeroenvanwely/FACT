import argparse

parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['ortho_lstm','vanilla_lstm','diversity_lstm'], required=True)
parser.add_argument('--attention', type=str, choices=['tanh', 'dot', 'all'], default='tanh')
parser.add_argument("--diversity",type=float,default=0)
parser.add_argument("--skip_experiments",action='store_true')
parser.add_argument("--skip_training",action='store_true')
parser.add_argument("--seed", type=int, default=None)

args, extras = parser.parse_known_args()
args.extras = extras

from Transparency.Trainers.DatasetQA import *
from Transparency.ExperimentsQA import *

# Function for setting the seed.
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU operation have separate seed.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

if args.seed:
    set_seed(args.seed)

dataset = datasets[args.dataset](args)
args.attention='tanh'

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

encoders = [args.encoder]

dataset.diversity = args.diversity
dataset.skip_training = args.skip_training
dataset.skip_experiments = args.skip_experiments
dataset.seed = args.seed

train_dataset_on_encoders(dataset, encoders)
generate_graphs_on_encoders(dataset, encoders)


