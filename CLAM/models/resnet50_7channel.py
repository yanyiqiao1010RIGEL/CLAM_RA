
import torch.nn as nn
import pretrainedmodels
def get_resnet50_(pretrained=True):
    model_name = 'resnet50_trunc'

    # Load pre-trained ResNet50 model
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    # Modify the first convolutional layer to accept 7 channels
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)

    # Copy pre-trained weights for RGB channels
    model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
    # Initialize the additional channels with the same weights as the first channel
    model.conv1.weight.data[:, 3:, :, :] = conv1.weight.data[:, :1, :, :]

    # Modify the fully connected layer to output a feature vector for each channel
    num_channels = 7  # Number of input channels
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adjust pooling layer to adapt to input size
    #print(model.last_linear.in_features)  #2048
    model.last_linear = nn.Linear(in_features=model.last_linear.in_features, out_features=num_channels * 262144, bias=True)
    return model
