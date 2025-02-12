from monai.networks.nets import UNet, SegResNet
from monai.inferers import sliding_window_inference
from torchinfo import summary
import torch
import torch.nn as nn

class UNetWithSigmoid(nn.Module):
    def __init__(self, base_model):
        super(UNetWithSigmoid, self).__init__()
        self.base_model = base_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.sigmoid(x)  # Apply Sigmoid at the end

def create_UNet3D(in_channels, out_channels, device, data_transform="", verbose=False):
    # TODO: finetune model parameters
    base_model = UNet(
        spatial_dims=3, # 3D
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=2
    ).to(device)

    # if data_transform == "SCALE_INTENSITY":
    model = UNetWithSigmoid(base_model)
    # else:
    #     model = base_model
        
    if verbose:
        # Calculate and display the total number of parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_params = count_parameters(model)
        print(f"Total number of trainable parameters: {total_params}")

        # Print the model architecture
        print(summary(model, input_size=(2, 3, 224, 224, 144), depth=6))

    return model

def create_SegResNet(in_channels, device):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=in_channels,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    return model

# define inference method
def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    with torch.amp.autocast('cuda'):
        return _compute(input)