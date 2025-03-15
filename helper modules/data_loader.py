"""
Contains functionality for creating DataLoaders from EMNIST dataset
"""
import torch, numpy as np, os
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import datasets

# batch size and num_workers
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

def create_dataloaders() -> tuple[DataLoader, DataLoader, np.ndarray, dict]:
  """Creates training and testing DataLoaders of the EMNIST dataset (for X and O only).
  Also, returns y_true as well as a dictionary mapping indices to labels (for X and 0)

  Returns
  -------
    train_dl: torch.utils.data.Dataloader
          Training DataLoader

    test_dl: torch.utils.data.Dataloader
          Testing DataLoader

    y_true: np.ndarray
          An ndarray of true labels from the test Subset that was used to create testing DataLoader

    label_map: dict
          A dictionary mapping index to classes
  """
  # class to warp around a Subset
  class set_wrapper(Dataset):
    def __init__(self, subset:Subset):
      self.subset = subset

    def __len__(self):
      return len(self.subset)

    def __getitem__(self, index):
      # get image and label at specified index
      x, y = self.subset[index]
      # transform label 15 -> 0 and 24 -> 1
      y = 0 if y==15 else 1
      return x, y

  # get the original dataset
  train_data = datasets.EMNIST(
      root='data', download=True, train=True, split='letters', transform=T.ToTensor())
  test_data = datasets.EMNIST(
      root='data', download=True, train=False, split='letters', transform=T.ToTensor())
  # label_map
  label_map = {0:'O', 1:'X'}

  # get indices for train and test data (where target==24 | target==15)
  train_indices = np.where((train_data.targets == 24) | (train_data.targets == 15))[0]
  test_indices = np.where((test_data.targets == 24) | (test_data.targets == 15))[0]
  np.random.shuffle(test_indices) # shuffle test_indices

  # from indices gotten above, create subsets for train and test
  train_set = Subset(dataset=train_data, indices=train_indices)
  test_set = Subset(dataset=test_data, indices=test_indices)

  # from the subsets above, wrap in custom set_wrapper class
  train_dataset, test_dataset = set_wrapper(train_set), set_wrapper(test_set)

  # get the targets from test Subset
  y_true = list()
  for x in range(len(test_set)):
    _, y = test_set[x]
    y_true.append(y)
  # turn targets (15 and 24 into 0 and 1 respectively)
  y_true = [0 if x==15 else 1 for x in y_true]
  # turn the targets list into a numpy array
  y_true = np.array(y_true)

  # create train and test DataLoaders from the Subsets created above
  train_dl = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)

  test_dl = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)

  # return
  return train_dl, test_dl, y_true, label_map
