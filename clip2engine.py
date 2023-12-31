import torch
import torch.onnx 
import onnx 
import tensorrt as trt 
import os

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
clip = model.cond_stage_model.transformer

clip_onnx_path = "clip.onnx"
clip_engine_path = "clip.engine"

input_names=['input_ids']
output_names=['text_embeddings', 'pooler_output']
dynamic_axes = { 
    'input_ids':[0], 
    'text_embeddings':[0,1],
    'pooler_output':[0]
}
clip_input = torch.zeros((1,77), dtype=torch.int32, device='cuda')

# if not os.path.isfile(clip_onnx_path):
torch.onnx.export( 
    clip, 
    clip_input, 
    clip_onnx_path,
    opset_version=17, 
    input_names=input_names, 
    output_names=output_names,
    # dynamic_axes=dynamic_axes,
    do_constant_folding=True,
)

# if not os.path.isfile(clip_engine_path):
# clip_onnx = onnx.load(clip_onnx_path)
# clip_engine = create_engine(clip_onnx)
# with open(clip_engine_path, mode='wb') as f: 
#     f.write(clip_engine)
#     print("generating file done!") 

os.system('polygraphy surgeon sanitize clip.onnx \
            --fold-constant \
            -o clip.onnx \
            > result-surgeon-clip.log')

#动态维度导出
# os.system("trtexec --onnx=clip.onnx --saveEngine=clip.engine --inputIOFormats=int32:chw --optShapes=input_ids:1x77")
os.system("trtexec --onnx=clip.onnx --saveEngine=clip.engine --inputIOFormats=int32:chw")



