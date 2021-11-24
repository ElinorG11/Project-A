""" The aihwkit does not support bit slicing. it generates a single tile large enough to contain all weights
    of the maximal layer (i.e., the layer with the maximum in*out channels). This test evaluates the effect
    of the tile size on the noisy output and how it deviates from the expected results. """

#    IMPORTS FROM TORCH    #
import torch
from torch import nn, cuda, device
import numpy as np
import matplotlib.pyplot as plt

#    IMPORTS FROM AIHWKIT    #
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog

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
def create_model(arch):
    """ Function to create the model. Uses basic pytorch libraries.
        The 1-layer model's are based on the maximal layer (in respect to in*out channels)
        structure of each architecture.
    :return: model
    """
    resnet = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                       bias=False)
    switcher = {
        'VGG': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        'MOBILENET': nn.Conv2d(in_channels=960, out_channels=960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               groups=1, bias=False)
    }
    return switcher.get(arch, resnet)


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


def generate_data(arch):
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
    mu, std = 0, np.sqrt(1/(2*np.pi))
    if arch == 'MOBILENET':
        size = (960, 960, 3, 3)
    else:
        size = (512, 512, 3, 3)
    x = np.random.normal(mu, std, size=size)

    # convert to tensor to move easily to GPU
    x = torch.from_numpy(x).float()

    return x


def store_output(ideal_output, analog_output):
    """ Function to store the output.
    :parameter ideal_output: the output of the ideal model
    :type ideal_output: FloatTensor
    :parameter analog_output: the output of the noisy model
    :type analog_output: FloatTensor
    :return
    """
    # store as tensor
    torch.save(ideal_output, 'ideal_output_tensor.pt')
    torch.save(analog_output, 'analog_output_tensor.pt')


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

    print("### Mean value is: " + str(errors_mean.item()) + " Standard deviation value is: " + str(errors_std.item()) + "\n")

    return errors_mean.item(), errors_std.item()


def plot_statistics(statistical_data):
    """ Function to visualize the distribution of the std and mean values of the noise effect by layer size
        :parameter: stat: statistics about std and mean of the difference between ideal and noisy output for
                          different layer sizes.
        :type: list of tuples (mean, std) of the error vector
        :returns:
    """
    m, s, x = [], [], np.zeros(len(statistical_data))
    for stats in statistical_data:
        m.append(stats[0])
        s.append(stats[1])

    # generate a vector of colors in the same length as the # of calculated statistics
    colors = np.random.rand(len(statistical_data))
    # 30 was chosen arbitrarily for the area radii, just so it'll be convenient to see the dots
    area = 30

    plt.figure(1)
    plt.title("Mean values of noise effect on the outputs for different architectures")
    plt.ylabel("Mean Value")
    plt.scatter(x, m, s=area, c=colors, alpha=0.5)
    plt.grid(color='green', linestyle='--', linewidth=0.5)

    plt.figure(2)
    plt.title("Std values of noise effect on the outputs for different architectures")
    plt.ylabel("Std Value")
    plt.scatter(x, s, s=area, c=colors, alpha=0.5)
    plt.grid(color='green', linestyle='--', linewidth=0.5)

    # TODO: need to add legend

    plt.show()


#    MAIN TEST     #
if __name__ == '__main__':

    # store tuples of the form (mean, std) for all the models in a list so we can later plot them
    stat = []

    for architecture in architectures:

        print("*************    using architecture " + architecture + "    *************")

        # create model
        model = create_model(architecture)

        print("### Ideal model structure:")
        print(model)

        # convert model to analog
        analog_model = conversion_to_analog(model)

        print("### Analog model structure:")
        print(analog_model)

        # generate input from normal dist
        data = generate_data(architecture)

        # move the model and data to cuda if it is available.
        """
        if use_cuda_device():
            model.cuda()
            analog_model.cuda()
            data = data.to(device_to_use)
        """

        # process data
        output_ideal = model(data)
        output_analog = analog_model(data)

        # store output
        store_output(output_ideal, output_analog)

        # calculate std and mean of the difference between ideal and noisy layer output
        stat.append(calc_statistics(output_ideal, output_analog))

    # plot the statistics for different layer sizes
    plot_statistics(stat)
