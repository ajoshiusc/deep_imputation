from monai.networks.nets import UNet, SegResNet
from monai.inferers import sliding_window_inference
import torch

def create_UNet3D(in_channels, out_channels, device, verbose=False):
    # TODO: finetune model parameters
    model = UNet(
        spatial_dims=3, # 3D
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=2
    ).to(device)
    # # Calculate and display the total number of parameters
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # total_params = count_parameters(model)
    # print(f"Total number of trainable parameters: {total_params}")

    # # Print the model architecture
    # print(f"Model Architecture:\n {model}")
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