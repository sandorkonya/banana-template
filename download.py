# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: a timm model

import timm

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = timm.create_model("resnet18", pretrained=True)

if __name__ == "__main__":
    download_model()
