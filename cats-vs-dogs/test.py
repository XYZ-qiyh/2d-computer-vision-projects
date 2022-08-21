import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
# from PIL import Image
import numpy as np
# import cv2

from net import Net
from dataset import TestDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'


def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def test():
    # model
    model = Net().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    for idx, sample in enumerate(test_dataloader):
        # print(sample)
        image, image_path = sample[0], sample[1]
        #print(type(image_path))
        # print(image_path[0].split(".")[0])
        image = tocuda(image)
        out = model(image)
        out = F.softmax(out, dim=1)
        out = out.cpu().detach().numpy().copy()
        label_id = np.argmax(out, axis=1)
        label = "cat" if label_id == 0 else "dog"
        #print(out)
        #print(label)
        print("{}: {}".format(image_path[0].split(".")[0], label))
        # break
    

if __name__ == "__main__":
    # data
    data_path = "/mnt/A/qiyh/2021/Dogs-vs-Cats/test1"
    test_dataset = TestDataset(data_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    # print(len(test_dataloader))

    # model
    # model_ckpt = "/mnt/A/qiyh/2021/Dogs-vs-Cats/DogsVsCats/model/model.pth"
    model_ckpt = "/mnt/A/qiyh/2021/Dogs-vs-Cats/code/ckpt/000009.pth"
    test()