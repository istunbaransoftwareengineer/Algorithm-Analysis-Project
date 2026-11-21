"""CIFAR-10 models used in the CPU benchmark."""

from torch import nn


def convolution(inputs, outputs, stride=1):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size=3, stride=stride, padding=1),
        nn.ReLU(),
    )


def mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 10),
    )


def small_cnn():
    return nn.Sequential(
        convolution(3, 8, stride=2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 10),
    )


def medium_cnn():
    return nn.Sequential(
        convolution(3, 12, stride=2),
        nn.MaxPool2d(2),
        convolution(12, 24),
        nn.MaxPool2d(2),
        convolution(24, 32),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 10),
    )


MODELS = {"MLP": mlp, "Small CNN": small_cnn, "Medium CNN": medium_cnn}
