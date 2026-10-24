import argparse
import pstats
import json
import torch
import os
import sys
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from card_arithmetic.dataset import CardArithmetic
from neurasp import NeurASP
from benchmark_device import resolve_device, synchronize
from card_arithmetic.network import CardNet
from mvpp import MVPP

parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=str, default='sum_2')
parser.add_argument('--image_dir', '--image-dir', type=str,
                    default=os.path.join(current_dir, 'images'),
                    help='Image directory (default: images/ beside this script)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', '--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--checkpoint_freq', type=int, default=100)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int)
parser.add_argument('--results-dir', default='results')
parser.add_argument('--no-cache', action='store_true',
                    help='Disable stable-model caching in training, validation, and testing')
devices = parser.add_mutually_exclusive_group()
devices.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
devices.add_argument('--cpu', action='store_true', help='Alias for --device cpu')
args = parser.parse_args()
try:
    device = resolve_device('cpu' if args.cpu else args.device)
except ValueError as error:
    parser.error(str(error))

# Set or configure seed for reproducibility
if args.seed is not None:
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
else:
    # We generate a random number as the seed, so that the experiment run can still be reproduced
    seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    random.seed(seed)

# Load Card arithmetic data for training, validation and testing
dataset = CardArithmetic(args.image_dir, args.variant)
val_dataset = CardArithmetic(args.image_dir, args.variant, 'val')
test_dataset = CardArithmetic(args.image_dir, args.variant, 'test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

network = CardNet().to(device)
with open(current_dir + '/tasks/card_facts.lp') as file:
    program = file.read()
with open(current_dir + f'/tasks/card_{args.variant}.lp') as file:
    program += '\n' + file.read()

# The latent concept for each card is a number
# neural_concepts = {'card': (int(args.variant.split('_')[1][0]), [f'{num}' for num in range(52)])}
neural_mapping = {'card': network}
optimizers = {'card': torch.optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)}
neurasp_obj = NeurASP(program, neural_mapping, optimizers, gpu=device.type == 'cuda')

synchronize(device)
neurasp_obj.learn(dataloader, args.epochs, lr=args.learning_rate, accStep = args.checkpoint_freq, seed = seed,
                  valDataset = val_dataloader, task = f'card_{args.variant}', storeSM=not args.no_cache)

synchronize(device)
dmvpp = MVPP(neurasp_obj.mvpp['program'])
down_acc, latent_accs = neurasp_obj.calculate_accuracies(test_dataloader, dmvpp, storeSM=not args.no_cache)
synchronize(device)
print(f"Downstream test accuracy: {down_acc * 100:.2f}%.")
results = {'algorithm': 'NeurASP', 'dataset': 'CardArithmetic', 'task': f'card_{args.variant}', 'seed': seed, 'epoch': args.epochs, 'step': 0,
           'batch_size': args.batch_size, 'downstream_test_accuracy': down_acc}
for concept in latent_accs:
    results[f'{concept}_lr'] = optimizers[concept].param_groups[0]['lr']
    results[f'{concept}_weight_decay'] = optimizers[concept].param_groups[0]['weight_decay']
    if latent_accs[concept] != 'unknown':
        print(f"Latent test accuracy for {concept}: {latent_accs[concept] * 100:.2f}%")
        results[f'{concept}_nn_test_accuracy'] = latent_accs[concept]

os.makedirs(args.results_dir, exist_ok=True)
with open(os.path.join(args.results_dir, f'card_{args.variant}_results.jsonl'), 'a') as f:
    f.write(json.dumps(results) + "\n")
