import torch
import numpy as np
import matplotlib.pyplot as plt

analog_output = torch.load("/home/elinor/aihwkit/test1_layer_size_noise/RESNET34/OUTPUT/analog_output.pth")
ideal_output = torch.load("/home/elinor/aihwkit/test1_layer_size_noise/RESNET34/OUTPUT/ideal_output.pth")

layers_analog_output = []
layers_ideal_output = []
output = []

for index in range(0,len(analog_output)-1,5):
  layer0_analog = analog_output[index+0].detach().numpy().reshape([1,-1])
  layer1_analog = analog_output[index+1].detach().numpy().reshape([1,-1])
  layer2_analog = analog_output[index+2].detach().numpy().reshape([1,-1])
  if index != 70:
    layer3_analog = analog_output[index+3].detach().numpy().reshape([1,-1])
    layer4_analog = analog_output[index+4].detach().numpy().reshape([1,-1])
  
  layers_analog_output.append([x for x in layer0_analog[0]])
  layers_analog_output.append([x for x in layer1_analog[0]])
  layers_analog_output.append([x for x in layer2_analog[0]])
  if index != 70:
    layers_analog_output.append([x for x in layer3_analog[0]])
    layers_analog_output.append([x for x in layer4_analog[0]])
  
  
  layer0_ideal = ideal_output[index+0].detach().numpy().reshape([1,-1])
  layer1_ideal = ideal_output[index+1].detach().numpy().reshape([1,-1])
  layer2_ideal = ideal_output[index+2].detach().numpy().reshape([1,-1])
  if index != 70:
    layer3_ideal = ideal_output[index+3].detach().numpy().reshape([1,-1])
    layer4_ideal = ideal_output[index+4].detach().numpy().reshape([1,-1])
  
  layers_ideal_output.append([x for x in layer0_ideal[0]])
  layers_ideal_output.append([x for x in layer1_ideal[0]])
  layers_ideal_output.append([x for x in layer2_ideal[0]])
  if index != 70:
    layers_ideal_output.append([x for x in layer3_ideal[0]])
    layers_ideal_output.append([x for x in layer4_ideal[0]])
  
  layers_analog_output = [item for sublist in layers_analog_output for item in sublist]
  layers_ideal_output = [item for sublist in layers_ideal_output for item in sublist]
  
  output.append(layers_analog_output)
  output.append(layers_ideal_output)
  
  plt.hist(output,bins=100)
  plt.legend(['Analog Output','Ideal Output'])
  
  path = "/home/elinor/aihwkit/minitest_output_distribution_resnet34/histogram_plots/"
  if index != 70:
    plt.savefig(path+"layers_"+str(index)+"-"+str(index+4)+".jpeg")
  else:
    plt.savefig(path+"layers_"+str(index)+"-"+str(index+2)+".jpeg")
  
  layers_analog_output.clear()
  layers_ideal_output.clear()
  output.clear()
  plt.clf()
  
  #plt.show()
