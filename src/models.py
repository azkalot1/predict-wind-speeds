import timm
import torch.nn as nn


class SimpleClassificationModel(nn.Module):
    def __init__(self, model_name='resnet34', n_input_channels=1, num_classes=1, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            num_classes=1,
            in_chans=n_input_channels,
            pretrained=pretrained)

    def forward(self, x):
        x = self.model(x)
        return(x)
