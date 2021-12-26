import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Imports from PyTorch.
from torchvision.models import resnet50

# Imports from aihwkit.
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaReRamSBPreset

# Device used in the RPU tile
RPU_CONFIG = TikiTakaReRamSBPreset()


def main():
    """Load a predefined model from pytorch library and convert to its analog version."""
    # Load the pytorch model.
    
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

    resnet50.eval().to(device)

    print(resnet50)
    
    uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
    ]
    
    batch = torch.cat(
        [utils.prepare_input_from_uri(uri) for uri in uris]
    ).to(device)
    
    with torch.no_grad():
      output = torch.nn.functional.softmax(resnet50(batch), dim=1)
    
    results = utils.pick_n_best(predictions=output, n=5)
    
    for uri, result in zip(uris, results):
      img = Image.open(requests.get(uri, stream=True).raw)
      img.thumbnail((256,256), Image.ANTIALIAS)
      plt.imshow(img)
      plt.show()
      print(result)
    
    model = resnet50

    # Convert the model to its analog version.
    model = convert_to_analog(model, RPU_CONFIG, weight_scaling_omega=0.6)

    print(model)


if __name__ == '__main__':
    # Execute only if run as the entry point into the program
    main()
