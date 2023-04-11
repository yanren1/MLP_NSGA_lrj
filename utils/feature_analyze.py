import os
import torch
import torch.nn as nn
import torchvision


def Convert_ONNX():
    model = torchvision.ops.MLP(in_channels=27,
                                # hidden_channels=[28, 32, 64, 128, 256, 128, 64, 32, 16, 8, 3],
                                hidden_channels=[28, 64, 256, 64, 8, 3],

                                # norm_layer=nn.LayerNorm,
                                dropout= 0,inplace=False)
    # set the model to inference mode
    # try read pre-train model
    weights_pth = 'final.pt'
    try:
        model.load_state_dict(torch.load(weights_pth,map_location='cpu'),)
    except:
        print(f'No {weights_pth}')

    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 27, requires_grad=True)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "final.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

if __name__ == '__main__':
    Convert_ONNX()