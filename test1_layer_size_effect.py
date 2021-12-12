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

from models import resnet, vgg, mobilenet

architectures = ['RESNET18', 'RESNET34', 'VGG', 'MOBILENET']

# moving data to GPU
device_to_use = device('cuda' if cuda.is_available() else 'cpu')


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

    path = "RESULTS/"
    if not os.path.exists(path):
        os.mkdir(path)

    path = "RESULTS/OUTPUTS"
    if not os.path.exists(path):
        os.mkdir(path)

    path = "RESULTS/OUTPUTS/" + arch
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

    #print("### Mean value is: " + str(errors_mean.item()) + " Standard deviation value is: " + str(errors_std.item()) + "\n")

    return errors_mean.item(), errors_std.item()


def plot_statistics_per_architecture(arch, statistical_data):
    """ Function to visualize the distribution of the std and mean values of the noise effect by layer size
        :parameter: stat: statistics about std and mean of the difference between ideal and noisy output for
                          different layer sizes.
        :type: list of tuples (mean, std) of the error vector
        :returns:
    """
    layer_size_list = get_layer_size(arch)

    m, s = [], []
    for stats in statistical_data:
        m.append(stats[0])
        s.append(stats[1])

    path = "RESULTS/OUTPUTS/" + arch
    if not os.path.exists(path):
        os.mkdir(path)
        
    plt.title("Noise effect by layer size for " + arch + " architecture")
    plt.ylabel("Mean Value")
    plt.xlabel("Layer Size")
    plt.ylim([-1, 1])
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.errorbar(layer_size_list, m, yerr=s, elinewidth=1, ecolor='red', fmt='o')
    plt.show()

    plt.savefig(path + '/statistical_plot_' + arch + '.png')

    m.clear()
    s.clear()


def plot_statistics_for_all_architectures():
    # load statistical data for all architectures
    stat_list, layer_size_list = [], []
    path = "RESULTS/STATISTICS/"
    for arch in architectures:
        stat_list.append(torch.load(path + arch + "/statistical_data.pth"))
        layer_size_list.append(get_layer_size(arch))
        
    path = "RESULTS/OUTPUTS/"
    if not os.path.exists(path):
        os.mkdir(path)
        

    # 'arch_m' and 'arch_s' will hold the statistics for all layer of a specific model
    # while 'm' and 's' will hold a list of the statistics for all models
    m, s = [], []
    # each 'statistics' is a list of tuples (mean, std) for each layer
    for arch_idx, arch in enumerate(architectures):
        # each 'statistics' is a list of tuples (mean, std) for each layer in a specific architecture
        for statistics in stat_list[arch_idx]:
            m.append(statistics[0])
            s.append(statistics[1])
       
        plt.title("Noise effect by layer size for different architectures")
        plt.ylabel("Mean Value")
        plt.xlabel("Layer Size")
        plt.ylim([-1, 1])
        plt.grid(color='green', linestyle='--', linewidth=0.5)
        plt.errorbar(layer_size_list[arch_idx], m, yerr=s, elinewidth=1, ecolor='red', fmt='o', label=arch)
        plt.legend(loc='best')
        m.clear()
        s.clear()
        
    plt.show()
    
    plt.savefig(path + '/statistical_plot_for_all_architectures.png')



#    MAIN TEST     #
if __name__ == '__main__':

    for architecture in architectures:

        # store tuples of the form (mean, std) for each layer for all the models in the list in order to plot them later
        stat = []

        print("*************    using architecture " + architecture + "    *************")

        # create model
        layers_list = create_single_layer_model(architecture)
        input_channels = get_channels(architecture)

        # generate input from normal dist
        data = generate_data(input_channels)

        layer_counter = 0

        for layer in layers_list:
            model = layer

            print("### Ideal model structure:")
            print(model)

            # convert model to analog
            analog_model = conversion_to_analog(model)

            print("### Analog model structure:")
            print(analog_model)

            # move the model and data to cuda if it is available.
            """
            if use_cuda_device():
                model.cuda()
                analog_model.cuda()
                data = data.to(device_to_use)
            """

            # process data for each layer

            # last linear layer needs some adaption for the input shape
            if layer == layers_list[-1] and (architecture == 'RESNET18' or architecture == 'RESNET34'):
                data = F.avg_pool2d(data, 4)
                data = data.view(data.size(0), -1)

            if layer == layers_list[-1] and architecture == 'VGG':
                data = data.view(data.size(0), -1)

            if layer == layers_list[-1] and architecture == 'MOBILENET':
                data = F.avg_pool2d(data, 2)
                data = data.view(data.size(0), -1)

            output_ideal = model(data)
            output_analog = analog_model(data)

            data = output_ideal

            # store output of each layer
            store_output(architecture, layer_counter, output_ideal, output_analog)

            # calculate std and mean of the difference between ideal and noisy layer output of each layer
            stat.append(calc_statistics(output_ideal, output_analog))

            layer_counter = layer_counter + 1

        # create directory to store statistics if needed
        directory = "RESULTS/STATISTICS"
        if not os.path.exists(directory):
            os.mkdir(directory)

        directory = "RESULTS/STATISTICS/" + architecture
        if not os.path.exists(directory):
            os.mkdir(directory)

        # store statistics per architecture
        torch.save(stat, directory + "/statistical_data.pth")

        # plot the statistics for different layer sizes
        plot_statistics_per_architecture(architecture, stat)

        stat.clear()

    plot_statistics_for_all_architectures()
