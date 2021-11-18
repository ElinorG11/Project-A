from torch import Tensor
from aihwkit.nn import AnalogLinear

# Test example to make sure I can activate the kit

model = AnalogLinear(2, 2)
model(Tensor([[0.1, 0.2], [0.3, 0.4]]))
