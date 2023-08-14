import torch
import torch.nn as nn

import os 
import sys

import cv2 
import numpy as np 
import requests 
import torch 
import torch.onnx 
from torch import nn 

from canny2image_torch import hackathon

import inspect
hk = hackathon()
hk.initialize()
model = hk.model

unet = model.model.diffusion_model
controlnet = model.control_model

# 创建一个并行模型
class CombinedModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.controlnet = model1
        self.unet = model2

    def forward(self, x_in, h_in, t_in, c_in):

        control = self.controlnet(x=x_in, hint=h_in, timesteps=t_in, context=c_in)
        
        eps = self.unet(x=x_in, timesteps=t_in, context=c_in, control=control)
        return eps

# 创建合并后的模型对象
combined_model = CombinedModel(controlnet, unet)

H = 256
W = 384

x_in = torch.randn(2, 4, H//8, W //8, dtype=torch.float32).to("cuda")
h_in = torch.randn(2, 3, H, W, dtype=torch.float32).to("cuda")
t_in = torch.zeros(2, dtype=torch.int32).to("cuda")
c_in = torch.randn(2, 77, 768, dtype=torch.float32).to("cuda")  

dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                    'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
                    't_in' : {0 : 'bs'},
                    'c_in' : {0 : 'bs'}}

# file_path = "combined_model.txt"
# with open(file_path, "w") as file:
#     file.write(str(combined_model))

with torch.inference_mode(), torch.autocast("cuda"):
    torch.onnx.export(combined_model,             
                        (x_in, h_in, t_in, c_in),  
                        "union_model.onnx", 
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        keep_initializers_as_inputs=True,
                        input_names = ['x_in', "h_in", "t_in", "c_in"], 
                        dynamic_axes = dynamic_table
                        )


os.system('polygraphy surgeon sanitize union_model.onnx --fold-constant  --save-external-data -o union_model_poly.onnx > result-surgeon-union_model.log')

os.system('trtexec --onnx=union_model_poly.onnx --saveEngine=union_model_poly.engine --fp16 --builderOptimizationLevel=3 --inputIOFormats=fp32:chw,fp32:chw,int32:chw,fp32:chw --optShapes=x_in:2x4x32x48,h_in:2x3x256x384,t_in:2,c_in:2x77x768')