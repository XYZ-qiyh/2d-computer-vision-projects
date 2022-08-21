from random import shuffle
from net import Net
from dataset import DogCatDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

## params
batch_size = 16
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
from tqdm import tqdm

checkpoints_dir = "./ckpt"
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def train():
    print("Hello World")
    dataset_path = "/mnt/A/qiyh/2021/Dogs-vs-Cats/TrainingData"
    train_dataset = DogCatDataset(dataset_path, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)

    
    dataset_path = "/mnt/A/qiyh/2021/Dogs-vs-Cats/TrainingData"

    model = Net()
    model = model.cuda()
    model = nn.DataParallel(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    criterion = torch.nn.CrossEntropyLoss()   

    cnt = 0
    n_epochs=10
    for epoch in range(n_epochs):
        #loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for img, label in train_dataloader:
            img, label = tocuda(img), tocuda(label)
            optimizer.zero_grad()
            # img = img.permute(0, 3, 1, 2)
            # print("input_img: {}".format(img.size()))
            out = model(img)
            loss = criterion(out, label.squeeze())
            loss.backward()
            optimizer.step()

            cnt += 1


            print("epoch: {}, idx: {}, train_loss: {}".format(epoch, cnt*batch_size, loss))

        torch.save(model.state_dict(), "./ckpt/{:0>6}.pth".format(epoch))







if __name__ == "__main__":
    train()