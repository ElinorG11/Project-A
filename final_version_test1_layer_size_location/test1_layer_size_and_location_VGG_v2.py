""" The aihwkit does not support bit slicing. it generates a single tile large enough to contain all weights
    of the maximal layer (i.e., the layer with the maximum in*out channels). This test evaluates the effect
    of the tile size on the noisy output and how it deviates from the expected results. """

#    IMPORTS FROM TORCH    #
import torch
from torch import nn, cuda, device
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt

#    IMPORTS FROM AIHWKIT    #
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import (
    AnalogSequential
)

from models import resnet, vgg, mobilenet

#architectures = ['RESNET18', 'RESNET34', 'VGG', 'MOBILENET']
architectures = ['VGG']

# moving data to GPU
device_to_use = device('cuda' if cuda.is_available() else 'cpu')
layer_count = 0
standard_dev = []

def use_cuda_device():
    """ Function to determine the device to fun on the model.
        we're using GPU Nvidia RTX 3070. We want to move the
        models and the data to this device """
    # Device to use
    use_cuda = 0
    if cuda.is_available():
        use_cuda = 1
    return use_cuda


#    HELPER METHODS     #
def create_single_layer_model(arch):
    """ Function to create the model. Uses basic pytorch libraries.
        The 1-layer model's are the original layers of the model.
    :return: single layer
    """
    resnet_list = resnet.get_layers()
    switcher = {
        'VGG': vgg.get_layers(),
        'MOBILENET': mobilenet.get_layers()
    }
    return switcher.get(arch, resnet_list)


def get_channels(arch):
    """ Function to return the # of input channels of the first layer in the model
        so we can determine the initial data shape.
    :return: number of channels
    :rtype: int
    """
    resnet_list = resnet.get_channels()
    switcher = {
        'VGG': vgg.get_channels(),
        'MOBILENET': mobilenet.get_channels()
    }
    return switcher.get(arch, resnet_list)


def get_layer_size(arch):
    resnet_list = resnet.get_layers_sizes()
    switcher = {
        'VGG': vgg.get_layers_sizes(),
        'MOBILENET': mobilenet.get_layers_sizes()
    }
    return switcher.get(arch, resnet_list)

def ProgramNoise(weights):
    '''
    Assuming IBM two-device per weight cell
    :param weights:
    :return:
    '''
    a = -1.1731
    b = 1.965
    c = 0.2635
    SigProg = a * torch.pow(weights, 2) + b * weights + c
    SigProg[SigProg < 0] = 0.
    Nprog = torch.distributions.normal.Normal(torch.zeros_like(weights), SigProg).sample()
    return Nprog

def add_noises(weights):
    MaxConductance = 25e-6
    W = weights
    MaxAbsWeight = torch.max(torch.abs(W))
    W = W * MaxConductance / MaxAbsWeight
    
    read_time = [0.3e3, 0.5e3, 1e3, 1.5e3, 2e3, 2.5e3, 1e4, 1.5e4, 5e4, 1e5, 1e6, 10e6, 100e6]
    DriftNoise = {'t_c': 1, 't_read': 250}
    x = np.arange(-10, 10, step=0.1)
    x = torch.from_numpy(x)

    W = W + MaxConductance * torch.sign(weights) * ProgramNoise(torch.abs(weights) / MaxConductance)
    
    for weight in W:
        
        weight = torch.tensor(weight)
        for i in range(10):
            mean_nu = torch.clamp(-0.0155 * (torch.abs(weight) / MaxConductance).log() + 0.0244, min=0.049, max=0.1)
            sigma_nu = torch.clamp(-0.0125 * (torch.abs(weight) / MaxConductance).log() - 0.0059, min=0.008, max=0.045)
            normal_dist = np.exp(-0.5 * ((x - mean_nu) / sigma_nu) ** 2) / (sigma_nu * math.sqrt(2 * math.pi))
            nu = torch.distributions.normal.Normal(mean_nu, sigma_nu).sample()
            time = torch.tensor(read_time)
            Gdrift = torch.mul(weight / MaxConductance, torch.pow(time / DriftNoise['t_c'], -nu))
            Q_s = torch.clamp(0.008 / torch.pow(torch.abs(weights) / MaxConductance, 0.65), min=0.2)
            sigma_nG = torch.mul(Gdrift, Q_s) * torch.sqrt(
                torch.log(torch.tensor((time + DriftNoise['t_read']) / (2 * DriftNoise['t_read']))))
            ReadNoise = torch.distributions.normal.Normal(torch.zeros_like(weights), sigma_nG).sample()
            
            Gdrift, ReadNoise = Gdrift * MaxConductance, torch.sign(weights) * ReadNoise * MaxConductance
            W = (Gdrift + ReadNoise)/scale
    return W
    
    
def conversion_to_analog(model_to_convert):
    """ Function to convert the model we've created into an analog
        model using the kit's built-in function. The kit's function
        convert_to_analog, automatically converts each layer into
        it's analog counterpart
    :param model_to_convert: the model to convert to analog
    :return analog model
    """
    # Define a single-layer network, using inference/hardware-aware training tile
    rpu_config = InferenceRPUConfig()

    # Inference noise model.
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

    # drift compensation
    rpu_config.drift_compensation = GlobalDriftCompensation()

    # Convert the model to its analog version.
    net = convert_to_analog(model_to_convert, rpu_config, weight_scaling_omega=0.6)
    
    net = AnalogSequential(net)
    
    net.eval()        # model needs to be in inference mode
    t_inference = 3600. # time of inference in seconds (after programming)

    net.program_analog_weights() # can also omitted as it is called below in any case
    net.drift_analog_weights(t_inference) # modifies weights according to noise model

    # now the model can be evaluated with programmed/drifted/compensated weights

    return net



def generate_data(in_channels):
    """ Function to generate fictive data. Since we want to stay true to real world
        data, our input will be generated from normal (Gaussian) distribution.
        Gaussian dist. models in high precision many real life phenomena, thus gives
        reliability to the affect of this data on our test.
    :return x: the data tensor
    :rtype x: torch FloatTensor
    """
    # create data from normal distribution with exp. value of 0 to simulate real life data.
    # :REMARKS:
    #   ++ array size is 60000 to simulate CIFAR10 dataset size
    #   ++ the std was chosen in such manner that the input samples will be bounded
    #       between [0,1] as if they went through normalization process.
    mu, std = 0, np.sqrt(1 / (2 * np.pi))
    size = (128, in_channels, 32, 32)
    x = np.random.normal(mu, std, size=size)

    # convert to tensor to move easily to GPU
    x = torch.from_numpy(x).float()

    return x


def store_output(arch, layer_number,ideal_output, analog_output):
    """ Function to store the output.
    :parameter ideal_output: the output of the ideal model
    :type ideal_output: FloatTensor
    :parameter analog_output: the output of the noisy model
    :type analog_output: FloatTensor
    :parameter layer_number: the number of the layer in a specific model
    :type layer_number: int
    :parameter arch: the architecture name
    :type arch: string
    :return
    """

    path = "MOBILENET/"
    if not os.path.exists(path):
        os.mkdir(path)

    path = "MOBILENET/OUTPUTS"
    if not os.path.exists(path):
        os.mkdir(path)

    # store as tensor
    torch.save(ideal_output, path + '/ideal_output_tensor_layer' + str(layer_number) + '.pt')
    torch.save(analog_output, path + '/analog_output_tensor_layer' + str(layer_number) + '.pt')


def calc_statistics(ideal_output, noisy_output):
    """ Function to calculate std and mean of the difference between ideal output and noisy output
        :parameter ideal_output: array of the outputs from the model when data is processed on ideal hw
        :type ideal_output tensor
        :parameter noisy_output: array of the outputs from the model when data is processed on noisy hw
                           which is simulated by aihwkit
        :type noisy_output tensor
        :returns statistics about std and mean of the difference between ideal and noisy output for
                  different layer sizes.
        :rtype tuple
    """
    errors = ideal_output - noisy_output
    errors_mean = torch.mean(errors)
    errors_std = torch.std(errors)
    
    global layer_count
    global standard_dev
    
    standard_dev.append(errors_std.item())

    print("### Mean value is: " + str(errors_mean.item()) + " Standard deviation value is: " + str(errors_std.item()) + " For layer number: " + str(layer_count) + "\n")
    
    layer_count = layer_count + 1

    return errors_mean.item(), errors_std.item()


def plot_statistics_by_layer_size(statistical_data, layer_size_list,layer_structure):
    """ Function to visualize the distribution of the std and mean values of the noise effect by layer size
        :parameter: stat: statistics about std and mean of the difference between ideal and noisy output for
                          different layer sizes.
        :type: list of tuples (mean, std) of the error vector
        :returns:
    """
    m, s = [], []
    for stats in statistical_data:
        m.append(stats[0])
        s.append(stats[1])

    path = "VGG"
    if not os.path.exists(path):
        os.mkdir(path)
        
    print("\n")
    print("==============================================>    STD")
    print(s)
    print("\n")
    print("==============================================>    MEAN")
    print(m)
    print("\n")
    print("==============================================>    LAYER SIZE")
    print(layer_size_list)

    plt.figure(1)
    plt.title("Noise effect by layer size for VGG16 architecture")
    plt.ylabel("Mean Value")
    plt.xlabel("Layer Size (in-channels * kernel-size)")
    plt.ylim([-0.25, 0.25])
    plt.xlim([0,2500])
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.xticks(layer_size_list,layer_structure,rotation=90)
    plt.errorbar(layer_size_list, m, yerr=s, elinewidth=2, ecolor='red', fmt='o', capsize=10)
   
    plt.savefig(path + '/statistical_plot_by_layer_size_vgg.png')
    
    plt.show()

    m.clear()
    s.clear()
    
def plot_statistics_by_layer_location(statistical_data):
    """ Function to visualize the distribution of the std and mean values of the noise effect by layer size
        :parameter: stat: statistics about std and mean of the difference between ideal and noisy output for
                          different layer sizes.
        :type: list of tuples (mean, std) of the error vector
        :returns:
    """
    layer_location_list = [x for x in range(len(statistical_data))]
    
    m, s = [], []
    for stats in statistical_data:
        m.append(stats[0])
        s.append(stats[1])

    path = "VGG"
    if not os.path.exists(path):
        os.mkdir(path)

    plt.figure(2)
    plt.title("Noise effect by layer size for VGG16 architecture")
    plt.ylabel("Mean Value")
    plt.xlabel("Layer Location")
    plt.ylim([-1, 1])
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.errorbar(layer_location_list, m, yerr=s, elinewidth=2, ecolor='red', fmt='o', capsize=10)
    
    plt.savefig(path + '/statistical_plot_by_layer_location_vgg.png')
    plt.show()

    m.clear()
    s.clear()


#    MAIN TEST     #
if __name__ == '__main__':

    model = vgg.VGG('VGG16')
    checkpoint = torch.load("/home/elinor/aihwkit/to_analog_backup/resnet18-cifar10-pytorch/checkpoint/""/home/elinor/aihwkit/to_analog_backup/trained weights/weights_vgg16.pth")
    #model.load_state_dict(checkpoint['net'])
    model = checkpoint['net']
    for key in model.keys():
      if key.find('weight') != -1:
        model[key] = add_noises(model[key])
    
    print(model)
    
    data = generate_data(3)
    outputs_ideal, outputs_analog, layers_sizes, layer_structure, stat = [], [], [], [], []

    for name, layer in model.named_children():
      if isinstance(layer, nn.Sequential):
        for l in layer:
          if isinstance(l, nn.Conv2d):
            print(l)
            out_ideal = l(data)
            
            layer_analog = conversion_to_analog(l)
            out_analog = layer_analog(data)
            
            outputs_ideal.append(out_ideal)
            outputs_analog.append(out_analog)
            layers_sizes.append(l.in_channels * l.kernel_size[0] * l.kernel_size[1])
            layer_structure.append("(" + str(l.in_channels)  + ", " + str(l.kernel_size[0]) + "x" + str(l.kernel_size[1]) + ")")
            
            stat.append(calc_statistics(out_ideal, out_analog))
            
            data = out_ideal
              
          else:
            out_ideal = l(data)
            
            layer_analog = conversion_to_analog(l)
            out_analog = layer_analog(data)
            
            outputs_ideal.append(out_ideal)
            outputs_analog.append(out_analog)
            
            data = out_ideal   
      elif isinstance(layer, nn.Linear):
        print(layer)
          
        data = data.view(-1, 512)
          
        out_ideal = layer(data)
        
        layer_analog = conversion_to_analog(layer)
        out_analog = layer_analog(data)
        
        outputs_ideal.append(out_ideal)
        outputs_analog.append(out_analog)
        
        layers_sizes.append(layer.in_features)
        layer_structure.append("(" + str(layer.in_features) + ")")
        stat.append(calc_statistics(out_ideal, out_analog))
      
        data = out_ideal   
        
        
    path = "VGG"
    if not os.path.exists(path):
        os.mkdir(path)
        
    path = "VGG/OUTPUT"
    if not os.path.exists(path):
        os.mkdir(path)
        
    diff_list = []
    for x,y in zip(outputs_ideal,outputs_analog):
      diff_list.append(abs(x-y))
    
    torch.save(outputs_ideal,path + "/ideal_output.pth")
    torch.save(outputs_analog,path + "/analog_output.pth")
    torch.save(diff_list,path + "/diff_output.pth")
    torch.save(layers_sizes,path + "/layer_sizes.pth")  
    torch.save(standard_dev,path + "/standard_dev.pth") 
    
    plot_statistics_by_layer_size(stat, layers_sizes,layer_structure)
    plot_statistics_by_layer_location(stat)
    #plot_statistics_per_architecture(stat,layers_sizes)
    
    
          
