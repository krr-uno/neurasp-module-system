import argparse
import os
from pathlib import Path
import random
import sys
import time

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR.parent))

import torch

from benchmark_device import resolve_device, synchronize
from benchmark_batching import collate_training_tuples

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch-size', '--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, help='Training random seed')
parser.add_argument('--no-cache', action='store_true',
                    help='Recompute stable models without reading or writing their cache')
args = parser.parse_args()
if args.epochs < 1 or args.batch_size < 1:
    parser.error('--epochs and --batch-size must be at least 1')
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
try:
    device = resolve_device(args.device)
except ValueError as error:
    parser.error(str(error))

# The baseline engine writes its cache relative to the working directory.
os.chdir(EXAMPLE_DIR)

from dataGen import dataList, obsList, testLoader
from network import Net
from neurasp import NeurASP
from benchmark_evaluation import evaluate_downstream

startTime = time.time()

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
# It can also be written in a file
######################################

dprogram = '''
nn(digit(4,i), [0,1,2,3,4,5,6,7,8,9]).
add2x2(R1,R2,C1,C2) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), digit(3,i,N4), R1=N1+N2, R2=N3+N4, C1=N1+N3, C2=N2+N4.
'''

########
# Define nnMapping and optimizers, initialze NeurASP object
########

m = Net().to(device)
nnMapping = {'digit': m}
optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}

NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=device.type == 'cuda')

dataset = list(zip(dataList, obsList))
if args.batch_size > 1:
    dataset = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_training_tuples)

########
# Start training and testing
########

# remove the saved models to fairly check total training time
if not args.no_cache:
    try:
        os.remove(EXAMPLE_DIR / 'saved_models/add2x2_stable_models.pkl')
    except FileNotFoundError:
        pass

synchronize(device)
time1 = time.time()
NeurASPobj.learn(dataset, epoch=args.epochs, bar=True, task='add2x2',
                 storeSM=not args.no_cache,
                 seed=args.seed if args.seed is not None else 'unknown')
synchronize(device)
print('--- total time for training: %s seconds ---' % int((time.time() - time1)) )
acc, _ = NeurASPobj.testNN('digit', testLoader)
print('Test Acc: {:0.2f}%'.format(acc))
evaluate_downstream(m, testLoader.dataset, 'add2x2', NeurASPobj.device)
synchronize(device)
print('--- total time from beginning: %s seconds ---' % int((time.time() - startTime)) )
