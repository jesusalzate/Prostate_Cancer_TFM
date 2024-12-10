import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBN


class EfficientNet_pretrained(nn.Module):
    """
    EfficientNet_pretrained: A class that utilizes a pretrained EfficientNet model 
    for 3D image classification, designed for grayscale 3D volumes (e.g., structural brain MRIs).

    The model uses an EfficientNet backbone for feature extraction and classification.

    Args:
        model_name (str): Model version/name to use from the EfficientNet variants. Defaults to "efficientnet-b0".
        n_classes (int): Number of output classes. Defaults to 2.
        in_channels_eff (int): Number of input channels for EfficientNet. Defaults to 1.
        pretrained_weights_path (str): Path to the pretrained weights of EfficientNet (Optional). Defaults to None.

    Forward Return:
        x (Tensor): The classification logits.

    Example:
        model = EfficientNet_pretrained(model_name='efficientnet-b0', n_classes=2, pretrained_weights_path=None)
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        n_classes: int = 2,
        in_channels_eff: int = 1,
        pretrained_weights_path: str = None,
    ):
        """
        Initialize the EfficientNet_pretrained model with the given parameters.

        Args:
            model_name (str): Model version/name to use from the EfficientNet variants.
            n_classes (int): Number of output classes.
            in_channels_eff (int): Number of input channels for EfficientNet.
            pretrained_weights_path (str): Path to the pretrained weights of EfficientNet (Optional).
        """
        super(EfficientNet_pretrained, self).__init__()

        # Instantiate the EfficientNet model
        EfficientNet = EfficientNetBN(
            model_name=model_name,
            pretrained=False,
            progress=False,
            spatial_dims=3,
            in_channels=in_channels_eff,
            num_classes=n_classes,
        )

        # Load pretrained weights into EfficientNet if provided
        if pretrained_weights_path:
            EfficientNet.load_state_dict(torch.load(pretrained_weights_path)["state_dict"])

        self.model = EfficientNet

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            x (Tensor): Classification logits of shape (batch_size, n_classes).
        """
        
        x = self.model(x)

        return x


class Bilinear3D(nn.Module):
    """
    Bilinear 3D architecture: siamese architecture with 3D convolutional layers.

    Designed for grayscale 3D volumes (i.e. estructural brain MRI)

    Args:
        n_classes (int): number of output classes (defaults to 2)
    """

    def __init__(self, n_classes: int = 2):
        super(Bilinear3D, self).__init__()
        self.conv3d_5_2 = nn.ModuleList(
            [nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(5, 5, 5), stride=(2, 2, 2), padding="valid") for _ in range(2)]
        )
        self.conv3d_3_1 = nn.ModuleList(
            [nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="valid") for _ in range(12)]
        )
        self.bn3d = nn.ModuleList([nn.BatchNorm3d(num_features=32) for _ in range(14)])
        self.avg = nn.ModuleList([nn.AvgPool3d(kernel_size=(2, 2, 2)) for _ in range(6)])
        self.dense_100 = nn.Linear(in_features=4096, out_features=100)
        self.dense = nn.Linear(in_features=100, out_features=n_classes)
        self.bn = nn.BatchNorm1d(num_features=100)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv3d_5_2[0](x)
        x1 = self.bn3d[0](x1)
        x1 = self.conv3d_3_1[0](x1)
        x1 = self.bn3d[1](x1)
        x1 = self.conv3d_3_1[1](x1)
        x1 = self.bn3d[2](x1)
        x1 = self.avg[0](x1)
        x1 = self.conv3d_3_1[2](x1)
        x1 = self.bn3d[3](x1)
        x1 = self.conv3d_3_1[3](x1)
        x1 = self.bn3d[4](x1)
        x1 = self.avg[1](x1)
        x1 = self.conv3d_3_1[4](x1)
        x1 = self.bn3d[5](x1)
        x1 = self.conv3d_3_1[5](x1)
        x1 = self.bn3d[6](x1)
        x1 = self.avg[2](x1)
        x1 = torch.flatten(x1, start_dim=1)

        x2 = self.conv3d_5_2[1](x)
        x2 = self.bn3d[7](x2)
        x2 = self.conv3d_3_1[6](x2)
        x2 = self.bn3d[8](x2)
        x2 = self.conv3d_3_1[7](x2)
        x2 = self.bn3d[9](x2)
        x2 = self.avg[3](x2)
        x2 = self.conv3d_3_1[8](x2)
        x2 = self.bn3d[10](x2)
        x2 = self.conv3d_3_1[9](x2)
        x2 = self.bn3d[11](x2)
        x2 = self.avg[4](x2)
        x2 = self.conv3d_3_1[10](x2)
        x2 = self.bn3d[12](x2)
        x2 = self.conv3d_3_1[11](x2)
        x2 = self.bn3d[13](x2)
        x2 = self.avg[5](x2)
        x2 = torch.flatten(x2, start_dim=1)

        x = torch.cat((x1, x2), dim=1)

        x = self.dense_100(x)
        x = self.bn(x)
        output = self.dense(x)
        #output = self.softmax(x)

        return output


def test():
    model = EfficientNet_pretrained(
        model_name="efficientnet-b7",
        n_classes=2,
        in_channels_eff=3,
        pretrained_weights_path=None,
    )
    # print(model)
    input = torch.randn(3, 3, 128, 128, 32)
    out = model(input)
    print(out.shape)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()



