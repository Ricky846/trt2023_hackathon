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
controlnet = model.control_model

controlnet_onnx_path = "controlnet.onnx"
controlnet_engine_path = "controlnet.engine"

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

torch.onnx.export(controlnet,               
                    (x_in, h_in, t_in, c_in),  
                    controlnet_onnx_path, 
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    keep_initializers_as_inputs=True,
                    input_names = ['x_in', "h_in", "t_in", "c_in"], 
                    output_names = output_names, 
                    dynamic_axes = dynamic_table)

# if not os.path.isfile(clip_onnx_path):


# if not os.path.isfile(clip_engine_path):
# controlnet_onnx = onnx.load(controlnet_onnx_path)
# controlnet_engine = create_engine(controlnet_onnx)
# with open(controlnet_engine_path, mode='wb') as f: 
#     f.write(controlnet_engine)
#     print("generating file done!") 

controlnet_onnx = onnx.load(controlnet_onnx_path)
controlnet_graph = gs.import_onnx(controlnet_onnx)
controlnet_graph = controlnet_graph.fold_constants()
controlnet_graph = controlnet_graph.cleanup()

onnx.save(gs.export_onnx(controlnet_graph), controlnet_onnx_path)



os.system("trtexec --onnx=controlnet.onnx --saveEngine=controlnet.engine --fp16 --inputIOFormats=fp32:chw,fp32:chw,int32:chw,fp32:chw --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")
