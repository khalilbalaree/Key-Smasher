import torch

def make2List(file):
    gen = []
    f = open(file,'r').readlines()
    for l in f:
        temp = l.replace('\n','')
        gen.append(list(temp.split(',')))
    
    return gen

def letterToIndex(letter):
    return 'abcdefghijklmnopqrstuvwxyz '.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, 27)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def keyToTensor(key):
    tensor = torch.tensor([letterToIndex(key)], dtype=torch.long)
    return tensor

def load_data():
    x = []
    y = []
    cut = 800
    fname = '../dataset/gen.txt'
    data = make2List(fname)
    for line in data:
        x.append([lineToTensor(line[0]), lineToTensor(line[1])])
        y.append(keyToTensor(line[2]))

    return x[:cut], y[:cut], x[cut:], y[cut:]

    