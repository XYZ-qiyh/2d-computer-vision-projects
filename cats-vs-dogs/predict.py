import torch
import torch.nn as nn
import torch.nn.functional as F
from net import Net

import os
import cv2
import numpy as np
from PIL import Image
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def predict(image_path=None):
    # model
    model = Net().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    image = Image.open(image_path)
    image = np.array(image, dtype=np.float32) / 255.0
    image = cv2.resize(image, (200, 200))
    input = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
    print(input.size())

    # inference
    out = model(input)
    #print(out.size())
    # print(out)

    # parse output
    out = F.softmax(out, dim=1)
    out = out.cpu().detach().numpy().copy()
    label_id = np.argmax(out, axis=1)
    label = "cat" if label_id == 0 else "dog"
    print("predict: {}".format(label))

    save_results(image_path=image_path, label=label)


def save_results(image_path, label):    
    '''
    Save Prediction Results
    '''
    if not os.path.exists("./results"):
        os.mkdir("./results")

    # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    in_image = cv2.imread(image_path)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (10, 120)
    # fontScale
    fontScale = 5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 3

    out_image = cv2.putText(in_image, label, org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)

    out_image_path = image_path.replace("data", "results")

    cv2.imwrite(out_image_path, out_image)

    
if __name__ == "__main__":
    # model ckpt
    model_ckpt = "./ckpt/000009.pth"

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    image_path = args.image_path

    # image_path = "./data/cat2.jpg"
    predict(image_path=image_path)