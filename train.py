from torch.optim import lr_scheduler
from rnn import MyModel
import torch
import torch.nn as nn
from tqdm import tqdm
from data import load_data
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, y_train, x_valid, y_valid = load_data()
N_train = len(x_train)
model = MyModel(device).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = lr_scheduler.StepLR(optimizer=optim, step_size=4, gamma=0.5)
batch_size = 1
criterion = nn.NLLLoss()

pbar = tqdm(range(20))
for epoch in pbar:
    model.train()
    total_loss = 0
    for b in range(int(np.ceil(N_train/batch_size)) ):  
        batch_x = x_train[b*batch_size : (b+1)*batch_size]
        batch_y = y_train[b*batch_size : (b+1)*batch_size]

        optim.zero_grad()
        output = model(batch_x[0][0].to(device), batch_x[0][1].to(device))
        loss = criterion(output, batch_y[0].to(device))
        total_loss += loss
        loss.backward()
        optim.step()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(len(x_valid)):
            output = model(x_valid[i][0].to(device), x_valid[i][1].to(device))
            _, predict = torch.max(output, dim=1)
            if predict.item() == y_valid[i].item():
                correct += 1
    pbar.set_description(str(total_loss / N_train) +' '+  str(correct/len(x_valid)))


            




    
    

torch.save(model.state_dict(), './checkpoint/mapper.pt')

# src = lineToTensor('this is me')
# targ = lineToTensor('not me')
# gt = keyToTensor('b')
# model = MyModel()
# o = model(src, targ)
# loss = criterion(o, gt)
# print(loss)

