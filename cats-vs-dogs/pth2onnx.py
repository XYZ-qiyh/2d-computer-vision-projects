import torch
import torch.nn as nn
import torch.onnx
from net import Net

def convert2onnx():
    # model
    model = Net()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_ckpt, weights_only=True))

    # set the model to inference mode
    model.train(False)
    model.cpu().eval()

    # An example input
    dummy_input = torch.rand(1, 3, 200, 200)
    torch.onnx.export(model.module, 
                      dummy_input, 
                      "exported_model.onnx",
                      opset_version=11)


if __name__ == "__main__":
    model_ckpt = "./ckpt/000009.pth"
    
    convert2onnx()