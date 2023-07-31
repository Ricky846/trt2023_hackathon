import cv2 
import numpy as np 
import requests 
import torch 
import torch.onnx 
from torch import nn 

from canny2image_TRT import hackathon

import inspect
import os
import sys

hk = hackathon()
hk.initialize()
model = hk.model

# 四个子模型
# self.state_dict = {
#     "clip": "cond_stage_model",
#     "control_net": "control_model",
#     "unet": "diffusion_model",
#     "vae": "first_stage_model"
# }

# "unet": "diffusion_model"
# 注意unet比较特殊，他是在self.model.model下面的一个diffusion_model属性里面
diffusion_model = model.model.diffusion_model

# "vae": "first_stage_model"
vae = model.first_stage_model

# "clip": "cond_stage_model"
# 因为clip调用了encode, encode调用了forward, forward输入时，一共就两步，一个是tokenizer,一个是transformer，
# transformer的__call__默认也是forward, 所以直接去clip的transformer导出onnx就行了
clip = model.cond_stage_model.transformer

# "control_net": "control_model",
control_model = model.control_model

device = 'cuda'

# 导出 clip 为 onnx 文件
if not os.path.isfile("clip.onnx"):
    dynamic_axes = { 
        'input_ids':[0], 
        'text_embeddings':[0,1],
        'pooler_output':[0]
    }
    clip_input = torch.rand(1,77).int().to(device)

    torch.onnx.export( 
        clip, 
        clip_input, 
        "clip.onnx",
        opset_version=17, 
        input_names=['input_ids'], 
        output_names=['text_embeddings', 'pooler_output'],
        dynamic_axes=dynamic_axes)

if not os.path.isfile("clip.engine"):
    os.system("trtexec --onnx=clip.onnx --saveEngine=clip.engine --fp16 --optShapes=input_ids:1x77")

# 导出 controlnet 为 onnx 文件
if not os.path.isfile("controlnet.onnx"):
    H = 256
    W = 384

    x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
    h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
    t_in = torch.zeros(1, dtype=torch.int32).to("cuda")
    c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")

    output_names = []
    for i in range(13):
        output_names.append("out_"+ str(i))

    dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                        'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
                        't_in' : {0 : 'bs'},
                        'c_in' : {0 : 'bs'}}

    for i in range(13):
        dynamic_table[output_names[i]] = {0 : "bs"}

    torch.onnx.export(control_model,               
                        (x_in, h_in, t_in, c_in),  
                        "controlnet.onnx", 
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        keep_initializers_as_inputs=True,
                        input_names = ['x_in', "h_in", "t_in", "c_in"], 
                        output_names = output_names, 
                        dynamic_axes = dynamic_table)
if not os.path.isfile("controlnet.engine"):
    os.system("trtexec --onnx=controlnet.onnx --saveEngine=controlnet.engine --fp16 --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")

# 导出 unet 为 onnx文件
if not os.path.isfile("unet.onnx"):
    diffusion_model_input1 = torch.zeros((1,4,32,48)).to("cuda")
    diffusion_model_input2 = torch.zeros(1, dtype=torch.int32).to("cuda")
    diffusion_model_input3 = torch.zeros((1,77,768)).to("cuda")

    diffusion_model_input4_1 = torch.zeros((1, 320, 32, 48)).to("cuda")
    diffusion_model_input4_2 = torch.zeros((1, 320, 32, 48)).to("cuda")
    diffusion_model_input4_3 = torch.zeros((1, 320, 32, 48)).to("cuda")

    diffusion_model_input4_4 = torch.zeros((1, 320, 16, 24)).to("cuda")

    diffusion_model_input4_5 = torch.zeros((1, 640, 16, 24)).to("cuda")
    diffusion_model_input4_6 = torch.zeros((1, 640, 16, 24)).to("cuda")

    diffusion_model_input4_7 = torch.zeros((1, 640, 8, 12)).to("cuda")

    diffusion_model_input4_8 = torch.zeros((1, 1280, 8, 12)).to("cuda")
    diffusion_model_input4_9 = torch.zeros((1, 1280, 8, 12)).to("cuda")

    diffusion_model_input4_10 = torch.zeros((1, 1280, 4, 6)).to("cuda")
    diffusion_model_input4_11 = torch.zeros((1, 1280, 4, 6)).to("cuda")
    diffusion_model_input4_12 = torch.zeros((1, 1280, 4, 6)).to("cuda")
    diffusion_model_input4_13 = torch.zeros((1, 1280, 4, 6)).to("cuda")

    input_names = ['sample', "timestep", "encoder_hidden_states"]
    control_input_names = []

    for i in range(1,14):
        input_names.append('control_input_' + str(i))
        control_input_names.append('control_input_' + str(i))

    # print(input_names)

    dynamic_table = {'sample' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 'encoder_hidden_states' : {0 : 'bs'}}

    for i in range(1,13):
        dynamic_table[control_input_names[i]] = {0 : 'bs', 2 : 'dim_2_' + str(i+1), 3: 'dim_3_' + str(i+1)}

    # print(dynamic_table)

    torch.onnx.export(diffusion_model,               
                        (diffusion_model_input1, diffusion_model_input2, diffusion_model_input3, 
                        [diffusion_model_input4_1, diffusion_model_input4_2, diffusion_model_input4_3, 
                        diffusion_model_input4_4, diffusion_model_input4_5, diffusion_model_input4_6, 
                        diffusion_model_input4_7, 
                        diffusion_model_input4_8, diffusion_model_input4_9, 
                        diffusion_model_input4_10, diffusion_model_input4_11, diffusion_model_input4_12, diffusion_model_input4_13]),
                        'unet.onnx',
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        # keep_initializers_as_inputs=True,
                        input_names = input_names, 
                        dynamic_axes = dynamic_table)
if not os.path.isfile("unet.engine"):
    os.system("trtexec --onnx=unet.onnx --saveEngine=unet.engine --inputIOFormats=fp32:chw,int32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw --optShapes=sample:1x4x32x48,timestep:1,encoder_hidden_states:1x77x768,control_input_1:1x320x32x48,control_input_2:1x320x32x48,control_input_3:1x320x32x48,control_input_4:1x320x16x24,control_input_5:1x640x16x24,control_input_6:1x640x16x24,control_input_7:1x640x8x12,control_input_8:1x1280x8x12,control_input_9:1x1280x8x12,control_input_10:1x1280x4x6,control_input_11:1x1280x4x6,control_input_12:1x1280x4x6,control_input_13:1x1280x4x6")

# 导出 vae 为 onnx 文件
if not os.path.isfile("vae.onnx"):
    vae_input = torch.rand(1,4,32,48).to(device)
    vae.forward = vae.decode
    torch.onnx.export(vae, vae_input, 'vae.onnx', 
                        input_names = ['v_in'], 
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,)
if not os.path.isfile("vae.engine"):
    os.system("trtexec --onnx=vae.onnx --fp16 --saveEngine=vae.engine")

print('finish')

