import numpy as np
import matplotlib.pyplot as plt


def plot_statistics(statistical_data):
    """ Function to visualize the distribution of the std and mean values of the noise effect by layer size
        :parameter: stat: statistics about std and mean of the difference between ideal and noisy output for
                          different layer sizes.
        :type: tuple
        :returns: Nothing
    """
    m, s, x = [], [], []
    for idx, stat in enumerate(statistical_data):
        x.append(idx)
        m.append(stat[0])
        s.append(stat[1])

    colors = np.random.rand(3)
    area = 30

    print(m)
    print(s)

    plt.scatter(x, m, s=area, c=colors, alpha=0.5)
    plt.show()

    plt.scatter(x, s, s=area, c=colors, alpha=0.5)
    plt.show()


if __name__ == '__main__':

    my_tuple_list = []

    for i in range(3):
        inp = (np.random.rand(), np.random.rand())
        my_tuple_list.append(inp)

    print(my_tuple_list)

    plot_statistics(my_tuple_list)
