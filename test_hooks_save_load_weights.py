import torch
from torch import nn, Tensor
# from torchvision.models import resnet18
from resnet import resnet32


# Add hooks.
class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def load_model(model, path):
    # Load.
    model.load_state_dict(torch.load(path))

    # Show model parameters.
    print(model.state_dict())


def main_load_weights():
    # Load the pytorch model.
    model = resnet32()

    # Specify the path to the parameters.
    path = "resnet32.th"

    # Apply model loading function.
    load_model(model, path)


def main_use_hooks():
    # Load the pytorch model.
    model = resnet32()

    # Attach Hooks.
    verbose_resnet = VerboseExecution(model)

    # Prepare dummy input.
    dummy_input = torch.ones(10, 3, 224, 224)

    _ = verbose_resnet(dummy_input)


if __name__ == '__main__':
    # Execute only if run as the entry point into the program
    main_load_weights()

