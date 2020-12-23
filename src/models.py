import timm
import torch.nn as nn


class SimpleClassificationModel(nn.Module):
    def __init__(self, model_name='resnet34', n_input_channels=1, num_classes=1, pretrained=True):
        super().__init__()
        m = timm.create_model(
            model_name,
            in_chans=n_input_channels,
            pretrained=pretrained)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nc, num_classes))

        self.features = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        )

    def get_features(self, x):
        x = self.enc(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.enc(x)
        x = self.head(x)
        return(x)
