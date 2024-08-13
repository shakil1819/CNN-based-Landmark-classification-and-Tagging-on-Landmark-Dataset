import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN architecture
class MyModel(nn.Module):
    """
    Defines a Convolutional Neural Network (CNN) architecture for image classification.

    Args:
        num_classes (int, optional): The number of output classes. Defaults to 1000.
        dropout (float, optional): The dropout rate for regularization. Defaults to 0.7.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Define the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an input tensor through the defined CNN architecture.

        Args:
            x (torch.Tensor): The input tensor of images.

        Returns:
            torch.Tensor: The output tensor of logits or probabilities.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, 1)  # Flatten the tensor

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)

def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    out = model(images)

    assert isinstance(out, torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"
    assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"
