import torch
from torch import nn
from torch.optim import SGD

def get_model(device:str) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
  """A function that returns a model, optimizer and loss function

  Parameters
  -------
    device: str
        The device on which to perform computation

  Returns
  -------
    model: torch.nn.Module
        A TinyVGG architecture model

    opt: torch.Optimizer
        An optimizer

    loss_fn: torch.nn.Module
        A loss function for multi-class classification
  """
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  # define model
  class TinyVGG(nn.Module):
    def __init__(self):
      super().__init__()
      # conv_block1
      self.conv_b1 = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 128, 3, padding=1),
          nn.BatchNorm2d(128),
          #nn.Dropout(p=0.15),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2))

      # conv_block2
      self.conv_b2 = nn.Sequential(
          nn.Conv2d(128, 256, 3, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 512, 3, padding=1),
          nn.BatchNorm2d(512),
          # nn.Dropout(p=0.15),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2))

      # classifier
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.BatchNorm1d(25088),
          nn.Linear(in_features=25088, out_features=256),
          nn.BatchNorm1d(256),
          nn.Dropout(p=0.05), # droput regularization
          nn.Linear(256, 2))

    def forward(self, x):
      return self.classifier(self.conv_b2(self.conv_b1(x)))

  # get an object of the model
  model = TinyVGG().to(device)

  # optimizer
  '''opt = torch.optim.Adam(params=model.parameters(),
                         lr=0.0009) # learning rate'''
  # optimizer
  opt = torch.optim.SGD(
      params=model.parameters(),
      lr=0.0001,  # learning rate
      momentum=0.5) #

  # loss_function
  loss_fn = nn.CrossEntropyLoss()

  return model, opt, loss_fn
