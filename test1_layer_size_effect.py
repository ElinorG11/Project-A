#    IMPORTS     #

#    HELPER METHODS     #
def create_model():
    """
    Function to create the model. Uses basic pytorch libraries.
    :return: model
    """
    raise NotImplementedError


def conversion_to_analog(model):
    """ Function to convert the model we've created into an analog
        model using the kit's built-in function. The kit's function
        convert_to_analog, automatically converts each layer into
        it's analog counterpart
    :param model: the model to convert to analog
    :return: analog model
    """
    raise NotImplementedError


def generate_data():
    """ Function to generate fictive data. Since we want to stay true to real world
        data, our input will be generated from normal (Gaussian) distribution.
        Gaussian dist. models in high precision many real life phenomena, thus gives
        reliability to the affect of this data on our test.
    :return nothing
    """
    raise NotImplementedError


def store_output(data):
    """ Function to store the output. Were storing the data in few ways: tensor,
        numpy array and json file to decide for later stages which format
        is the most convenient for use to work with
    :parameter data: the output of the model
    :type tensor
    :return
    """
    raise NotImplementedError


def calc_statistics(ideal, noisy):
    """ Function to calculate std and mean of the difference between ideal output and noisy output
        :parameter ideal: array of the outputs from the model when data is processed on ideal hw
        :type tensor
        :parameter noisy: array of the outputs from the model when data is processed on noisy hw
                           which is simulated by aihwkit
        :type tensor
        :returns statistics about std and mean of the difference between ideal and noisy output for
                  different layer sizes.
        :rtype tuple
    """
    raise NotImplementedError


def plot_statistics(stat):
    """ Function to visualize the distribution of the std and mean values of the noise effect by layer size
        :parameter: stat: statistics about std and mean of the difference between ideal and noisy output for
                          different layer sizes.
        :type: tuple
        :returns: Nothing
    """
    raise NotImplementedError


#    MAIN TEST     #
if __name__ == '__main__':
    # create model
    model = create_model()

    # convert model to anaglog
    analog_model = conversion_to_analog(model)

    # generate input from normal dist
    data = generate_data()

    # process data
    output_ideal = model(data)
    output_analog = analog_model(data)

    # store output
    store_output(data)

    # calculate std and mean of the difference between ideal and noisy layer output
    stat = calc_statistics(output_ideal, output_analog)

    # plot the statistics for different layer sizes
    plot_statistics(stat)
