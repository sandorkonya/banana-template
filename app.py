import timm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T,datasets
import torch.nn.functional as F 
from io import BytesIO
import base64

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global img_transform
    
    img_transform = T.Compose([
                             T.Resize(size=(384,384)), # Resizing the image to be 384 x 384
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
    ])
    
    device = 0 if torch.cuda.is_available() else -1
    model = timm.create_model("hf_hub:timm/mobilenetv3_large_100.ra_in1k", pretrained=True) 

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global img_transform

    # Parse out your arguments
    imagedata = model_inputs.get('imagedata', None)
    if imagedata == None:
        return {'message': "No imagedata provided"}
    
    # Assuming imagedata is the string value with 'data:image/jpeg;base64,' we remove the first 23 char
    image = Image.open(BytesIO(base64.decodebytes(bytes(imagedata[23:], "utf-8"))))
    image = img_transform(image)
    ps = model(image.to("cpu").unsqueeze(0))
    ps = F.softmax(ps,dim = 1)
    result = ps.cpu().data.numpy()[0]
    
    # Return the results as a dictionary
    return result
