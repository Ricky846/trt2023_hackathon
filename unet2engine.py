import torch
import torch.onnx 
import onnx 
import tensorrt as trt 
import os

import onnx_graphsurgeon as gs
from canny2image_torch import hackathon

def create_engine(onnx_model):
    logger = trt.Logger(trt.Logger.ERROR) 
    builder = trt.Builder(logger) 
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    config = builder.create_builder_config() 
    # 设置TensorRT工作空间大小为1GB
    config.max_workspace_size  = 4<<30 
    profile = builder.create_optimization_profile() 
    # parse onnx 
    parser = trt.OnnxParser(network, logger) 
    if not parser.parse(onnx_model.SerializeToString()): 
        error_msgs = '' 
        for error in range(parser.num_errors): 
            error_msgs += f'{parser.get_error(error)}\n' 
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 

    # 获取 onnx 文件中的输入和输出张量的名称和形状
    input_list = []
    output_list = []

    # 设定输入，输入张量的名称、形状的最小值和最大值。
    for i in range(len(onnx_model.graph.input)):
        input_name = onnx_model.graph.input[i].name
        print(onnx_model.graph.input[i])
        input_shape = tuple(d.dim_value for d in onnx_model.graph.input[i].type.tensor_type.shape.dim)
        input_shape = tuple(1 if x == 0 else x for x in input_shape)
        print(input_name)
        print(input_shape)
        input_list.append([input_name, input_shape])
        profile.set_shape(input_name, input_shape, input_shape, input_shape) 


    for i in range(len(onnx_model.graph.output)):
        output_name = onnx_model.graph.output[i].name
        # print(onnx_model.graph.output[i])
        output_shape = tuple(d.dim_value for d in onnx_model.graph.output[i].type.tensor_type.shape.dim)
        print(output_name)
        print(output_shape)
        output_list.append([output_name, output_shape])
        profile.set_shape(output_name, output_shape, output_shape, output_shape) 

    # 设定输入，输入张量的名称、形状的最小值和最大值。
    # profile.set_shape('input', [1,4,32,48], [1,4,32,48], [1,4,32,48]) 
    config.add_optimization_profile(profile) 
    # create engine 
    device = torch.device('cuda:0') 
    with torch.cuda.device(device): 
        engine = builder.build_serialized_network(network, config)

    return engine


hk = hackathon()
hk.initialize()
model = hk.model
unet = model.model.diffusion_model

unet_onnx_path = "unet_temp.onnx"
# unet_engine_path = "unet.engine"

diffusion_model_input1 = torch.zeros((1,4,32,48), dtype=torch.float32, device='cuda')
diffusion_model_input2 = torch.zeros(1, dtype=torch.int32, device='cuda')
diffusion_model_input3 = torch.zeros((1,77,768), dtype=torch.float32, device='cuda')

diffusion_model_input4_1 = torch.zeros((1, 320, 32, 48), dtype=torch.float32, device='cuda')
diffusion_model_input4_2 = torch.zeros((1, 320, 32, 48), dtype=torch.float32, device='cuda')
diffusion_model_input4_3 = torch.zeros((1, 320, 32, 48), dtype=torch.float32, device='cuda')

diffusion_model_input4_4 = torch.zeros((1, 320, 16, 24), dtype=torch.float32, device='cuda')

diffusion_model_input4_5 = torch.zeros((1, 640, 16, 24), dtype=torch.float32, device='cuda')
diffusion_model_input4_6 = torch.zeros((1, 640, 16, 24), dtype=torch.float32, device='cuda')

diffusion_model_input4_7 = torch.zeros((1, 640, 8, 12), dtype=torch.float32, device='cuda')

diffusion_model_input4_8 = torch.zeros((1, 1280, 8, 12), dtype=torch.float32, device='cuda')
diffusion_model_input4_9 = torch.zeros((1, 1280, 8, 12), dtype=torch.float32, device='cuda')

diffusion_model_input4_10 = torch.zeros((1, 1280, 4, 6), dtype=torch.float32, device='cuda')
diffusion_model_input4_11 = torch.zeros((1, 1280, 4, 6), dtype=torch.float32, device='cuda')
diffusion_model_input4_12 = torch.zeros((1, 1280, 4, 6), dtype=torch.float32, device='cuda')
diffusion_model_input4_13 = torch.zeros((1, 1280, 4, 6), dtype=torch.float32, device='cuda')

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

torch.onnx.export(unet,               
                    (diffusion_model_input1, diffusion_model_input2, diffusion_model_input3, 
                    [diffusion_model_input4_1, diffusion_model_input4_2, diffusion_model_input4_3, 
                    diffusion_model_input4_4, diffusion_model_input4_5, diffusion_model_input4_6, 
                    diffusion_model_input4_7, 
                    diffusion_model_input4_8, diffusion_model_input4_9, 
                    diffusion_model_input4_10, diffusion_model_input4_11, diffusion_model_input4_12, diffusion_model_input4_13]),
                    unet_onnx_path,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    # keep_initializers_as_inputs=True,
                    input_names = input_names, 
                    dynamic_axes = dynamic_table)

# net_onnx = onnx.load(unet_onnx_path)
# net_graph = gs.import_onnx(net_onnx)
# net_graph = net_graph.fold_constants()
# net_graph = net_graph.cleanup()
# net_onnx = gs.export_onnx(net_graph)

# onnx.save(net_onnx, "unet.onnx", save_as_external_data=True)

# 使用 polygraphy 工具对网络的节点进行折叠
os.system('polygraphy surgeon sanitize unet_temp.onnx \
            --fold-constant \
            -o unet.onnx \
            --save-external-data \
            > result-surgeon.log')

os.system("trtexec --onnx=unet.onnx --saveEngine=unet.engine --fp16 --builderOptimizationLevel=5 --inputIOFormats=fp32:chw,int32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw --optShapes=sample:1x4x32x48,timestep:1,encoder_hidden_states:1x77x768,control_input_1:1x320x32x48,control_input_2:1x320x32x48,control_input_3:1x320x32x48,control_input_4:1x320x16x24,control_input_5:1x640x16x24,control_input_6:1x640x16x24,control_input_7:1x640x8x12,control_input_8:1x1280x8x12,control_input_9:1x1280x8x12,control_input_10:1x1280x4x6,control_input_11:1x1280x4x6,control_input_12:1x1280x4x6,control_input_13:1x1280x4x6")