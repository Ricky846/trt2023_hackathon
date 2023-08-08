from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import tensorrt as trt
from cuda import cudart

import os

torch_dtype = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool
}

class model_engine():
    def __init__(self, context, input_tensor, output_tensor):
        self.context = context
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

class hackathon():

    def initialize(self):
        os.system("sed -i 's/use_checkpoint: True/use_checkpoint: False/g' '/home/player/ControlNet/models/cldm_v15.yaml'")

        self.apply_canny = CannyDetector()
        self.model = create_model('/home/player/ControlNet/models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        
        # if(os.path.isfile("/home/player/ControlNet/models/engine/clip.engine") and os.path.isfile("/home/player/ControlNet/models/engine/controlnet.engine") and os.path.isfile("/home/player/ControlNet/models/engine/unet.engine") and os.path.isfile("/home/player/ControlNet/models/engine/vae.engine")):
        # trt 初始化
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        # 加载plugin
        trt.init_libnvinfer_plugins(self.trt_logger, '')

        self.model.cond_stage_model.run_engine = self.run_engine
        self.model.run_engine = self.run_engine
        # self.model.decode_first_stage.run_engine = self.run_engine

        # 初始化 engine
        # self.model.cond_stage_model.clip_engine = self.load_engine('clip')
        self.model.controlnet_engine = self.load_engine('control_net')
        self.model.unet_engine = self.load_engine('unet')
        self.model.vae_engine = self.load_engine('vae')

    def load_engine(self, engine_name):
        # 加载 cond_stage_model : clip
        if(engine_name == 'clip'):
            engine_path = 'clip.engine'
            with open(engine_path, mode='rb') as f: 
                                engine_data = f.read()
        elif(engine_name == 'control_net'):
            engine_path = 'controlnet.engine'
            with open(engine_path, mode='rb') as f: 
                                engine_data = f.read()
        elif(engine_name == 'unet'):
            engine_path = 'unet.engine'
            with open(engine_path, mode='rb') as f: 
                                engine_data = f.read()
        elif(engine_name == 'vae'):
            engine_path = 'vae.engine'
            with open(engine_path, mode='rb') as f: 
                                engine_data = f.read()

        # 创建 engine
        engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        # 获取输入输出的数量以及name
        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        nOutput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)

        input_tensor = []
        output_tensor = []
        # 绑定输入的shape,并将输入的tensor的信息保存
        for i in range(nInput):
            input_name = lTensorName[i]
            input_shape = engine.get_tensor_shape(lTensorName[i])
            input_dtype = engine.get_tensor_dtype(lTensorName[i])
            input_tensor.append({'name' : input_name, 'shape' : input_shape, 'dtype' : input_dtype})
            context.set_input_shape(lTensorName[i], engine.get_tensor_shape(lTensorName[i]))
        # 将输出的tensor的信息保存
        for i in range(nOutput):
            output_name = lTensorName[nInput + i]
            output_shape = engine.get_tensor_shape(output_name)
            output_dtype = engine.get_tensor_dtype(output_name)
            output_tensor.append({'name' : output_name, 'shape' : output_shape, 'dtype' : output_dtype})

        engine = model_engine(context, input_tensor, output_tensor)
        return engine
    
    def run_engine(self, engine, input_datas):
        nInput = len(engine.input_tensor)
        # nOutput = len(engine.output_tensor)
        # 将输入拷贝到gpu
        # 通过shape判断输入
        # for tensor in engine.input_tensor:
        #     for data in input_datas:
        #         if(data.shape == tensor['shape']):
        #             # print(tensor['name']+' : '+str(tensor['shape']))
        #             data = data.cpu().reshape(-1)
        #             inputHost = np.ascontiguousarray(data)
        #             inputDevice = cudart.cudaMalloc(inputHost.nbytes)[1]
        #             engine.context.set_tensor_address(tensor['name'], inputDevice)
        #             cudart.cudaMemcpy(inputDevice, inputHost.ctypes.data, inputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        #             break
        #         else:
        #             continue
        # 不进行判断，直接按顺序将输入拷贝到cuda
        # inputDevice_list = []
        for i in range(nInput):
            # print(engine.input_tensor[i]['name']+' : '+str(engine.input_tensor[i]['shape']))
            # print('input data shape : ' + str(input_datas[i].shape))
            # data = input_datas[i].cpu().reshape(-1)
            # inputHost = np.ascontiguousarray(data)
            # inputDevice = cudart.cudaMalloc(inputHost.nbytes)[1]
            # engine.context.set_tensor_address(engine.input_tensor[i]['name'], inputDevice)
            # cudart.cudaMemcpy(inputDevice, inputHost.ctypes.data, inputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            # inputDevice_list.append(inputDevice)
            data = input_datas[i].reshape(-1)
            inputDevice = data.data_ptr()
            engine.context.set_tensor_address(engine.input_tensor[i]['name'], inputDevice)
            # inputDevice_list.append(inputDevice)
            
        # 为输出分配地址并保存地址信息
        # outputDevice_list = []
        outputs = []
        for tensor in engine.output_tensor:
            # outputHost = np.empty(tensor['shape'], trt.nptype(tensor['dtype']))
            # print(tensor['shape'])
            # print(tensor['dtype'])
            # print(torch_dtype[tensor['dtype']])
            output = torch.zeros(tuple(tensor['shape']), dtype=torch_dtype[tensor['dtype']], device='cuda')
            outputDevice = output.reshape(-1).data_ptr()
            # outputDevice = cudart.cudaMalloc(outputHost.nbytes)[1] 
            engine.context.set_tensor_address(tensor['name'], outputDevice)
            # outputDevice_list.append(outputDevice)
            outputs.append(output)

        engine.context.execute_async_v3(0)

        # 将输出拷贝出来
        # outputs = []
        # for i in range(nOutput):
        #     outputDevice = outputDevice_list[i]
        #     # outputHost = np.empty(engine.output_tensor[i]['shape'], trt.nptype(engine.output_tensor[i]['dtype']))
        #     # cudart.cudaMemcpy(outputHost.ctypes.data, outputDevice, outputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        #     # outputs.append(outputHost)
        #     outputs.append(torch.cuda.memory_allocated().from_address(outputDevice))
        #     cudart.cudaFree(outputDevice)
        # for i in range(nInput):
        #     cudart.cudaFree(inputDevice_list[i])

        return outputs

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        
        ddim_steps = 8
        with torch.no_grad():
            # 对图片进行预处理
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)


            # 原始代码
            # 处理描述词
            # cond_stage_model : clip
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            # c_concat 调用 canny 算子得到的图像边缘分割的结果
            # c_concat = cond['c_concat']
            # print(len(c_concat))
            # print("c_concat shape: ", c_concat[0].shape)
            # c_crossattn 描述词转换的结果
            # c_crossattn = cond['c_crossattn']
            # print(len(c_crossattn))
            # print("c_crossattn shape: ", c_crossattn[0].shape)
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # vae 输入 torch.Size([1, 4, 32, 48])
            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results
