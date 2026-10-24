import os
import argparse
import random
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from benchmark_device import resolve_device, synchronize
from benchmark_batching import collate_training_tuples

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', '--batch-size', type=int, default=1)
parser.add_argument('--seed', type=int, help='Training random seed')
parser.add_argument('--no-cache', action='store_true',
                    help='Disable stable-model caching (already disabled for this example)')
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

from network import Net, testNN
from neurasp import NeurASP
from benchmark_evaluation import evaluate_downstream

start_time = time.time()

#############################
# Construct the training and testing data
#############################

class MNIST_Addition(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return self.dataset[i1][0], self.dataset[i2][0], l

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
# used for training
trainDataset = MNIST_Addition(torchvision.datasets.MNIST(root=dir_path+'/../data/', train=True, download=True, transform=transform), dir_path+'/../data/mnistAdd_train.txt')
# used to check training and testing accuracy
testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root=dir_path+'/../data/', train=False, transform=transform), batch_size=1000)
trainLoader = torch.utils.data.DataLoader(Subset(torchvision.datasets.MNIST(root=dir_path+'/../data/', train=True, transform=transform), range(1000)), batch_size=1000)

dataList = []
obsList = []
for i1, i2, l in trainDataset:
    dataList.append({'i1': i1.unsqueeze(0), 'i2': i2.unsqueeze(0)})
    obsList.append(':- not addition(i1, i2, {}).'.format(l))


#############################
# NeurASP program
#############################

dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
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

print(f'Start training for {args.epochs} epochs (batch size {args.batch_size})...')
synchronize(device)
NeurASPobj.learn(dataset, epoch=args.epochs, storeSM=False, bar=True, task='mnistAdd',
                 seed=args.seed if args.seed is not None else 'unknown')

synchronize(device)
device = NeurASPobj.device
# check testing accuracy
accuracy, singleAccuracy = testNN(model=m, testLoader=testLoader, device=device)
# check training accuracy
accuracyTrain, singleAccuracyTrain = testNN(model=m, testLoader=trainLoader, device=device)
print(f'{accuracyTrain:0.2f}\t{accuracy:0.2f}')
evaluate_downstream(m, testLoader.dataset, 'mnistAdd', device)
synchronize(device)
print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time) )
