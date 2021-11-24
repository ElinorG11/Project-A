import torch
import json


def example1():
    # create a FloatTensor
    x = torch.zeros([2, 4], dtype=torch.float32)

    # convert to dictionary. json format accepts only serializable objects
    my_dict = {'data': []}
    i = 0
    for item in x:
        i = i + 1
        label = 'label' + str(i)
        my_dict['data'].append({'label': item.tolist()})

    print(my_dict)

    # store in json format
    with open("my_data.json", "w") as f:
        json.dump(my_dict, f)


def example2():
    in_dict = {"train": [{"input": [[3, 1, 2], [3, 1, 2], [3, 1, 2]], "output": [[4, 5, 6], [4, 5, 6], [4, 5, 6]]},
                         {"input": [[2, 3, 8], [2, 3, 8], [2, 3, 8]], "output": [[6, 4, 9], [6, 4, 9], [6, 4, 9]]}]}

    train_examples = []
    for item in in_dict['train']:
        in_tensor = torch.Tensor(item['input'])
        out_tensor = torch.Tensor(item['output'])
        train_examples.append([in_tensor, out_tensor])

    out_dict = {'train': []}
    for item in train_examples:
        out_dict['train'].append({
            'input': item[0].tolist(),
            'output': item[1].tolist()
        })

    print(out_dict)


# load from json and convert back from dict to tensor
def example3():
    my_dict = json.load(open('my_data.json'))

    print(my_dict)

    x_tensor = []
    i = 0
    for item in my_dict['data']:
        i = i + 1
        label = 'label' + str(i)
        x_tensor = torch.Tensor(item['label'])

    print(x_tensor, x_tensor.type())


if __name__ == '__main__':
    example1()
    print("\n")
    example2()
    print("\n")
    example3()
