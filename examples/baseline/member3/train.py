import os
import argparse
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

import torch

from benchmark_device import resolve_device, synchronize
from benchmark_batching import collate_training_tuples

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', '--batch-size', type=int, default=1)
parser.add_argument('--seed', type=int, help='Training random seed')
parser.add_argument('--no-cache', action='store_true',
                    help='Recompute stable models for every observation; do not read or write their cache')
args = parser.parse_args()
if args.epochs < 1 or args.batch_size < 1:
    parser.error("--epochs and --batch-size must be at least 1")
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
try:
    device = resolve_device(args.device)
except ValueError as error:
    parser.error(str(error))

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
nn(digit(3,i), [0,1,2,3,4,5,6,7,8,9]).
member(D,0) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), 
               check(D), D!=N1, D!=N2, D!=N3.
member(D,1) :- check(D), not member(D,0).
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
        os.remove('saved_models/member3_stable_models.pkl')
    except OSError:
        pass

synchronize(device)
time1 = time.time()
NeurASPobj.learn(dataset, epoch=args.epochs, bar=True, task='member3', storeSM=not args.no_cache,
                 seed=args.seed if args.seed is not None else 'unknown')
synchronize(device)
print('--- total time for training: %s seconds ---' % int((time.time() - time1)) )
acc, _ = NeurASPobj.testNN('digit', testLoader)
print('Test Acc: {:0.2f}%'.format(acc))
evaluate_downstream(m, testLoader.dataset, 'member3', NeurASPobj.device)
synchronize(device)
print('--- total time from beginning: %s seconds ---' % int((time.time() - startTime)) )
