import torch, numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

# del torch, F, f1_score, accuracy_score

# function for model training
def train_batches(model:torch.nn.Module, train_dl:torch.utils.data.DataLoader,
                optimizer:torch.optim.Optimizer, loss_fn:torch.nn.Module, device:str) -> tuple[float, float, float]:
  """Trains model on all batches of train-set DataLoader and returns
      average training loss, accuracy and f1_score

  Parameters
  -------
    model: torch.nn.Module
        The model being trained

    train_dl: torch.utils.data.DataLoader
        DataLoader for training data

    optimizer: torch.optim.Optimizer
        The optimizer

    loss_fn: torch.nn.Module
        Function used to calculate loss

    device: str
        The device on which computation occurs

  Returns
  -------
    ls: float
        average test loss across all batches of data
    acc: float
        average test accuracy across all batches of data
    f1: float
        average test f1_score across all batches of data
  """
  # for reproducability
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  ls, acc, f1 = 0, 0, 0

  #training mode
  model.train()

  for x, y in train_dl:
    # move x, y to device
    x, y = x.to(device), y.to(device)
    # zero_grad
    optimizer.zero_grad()

    # forward pass
    logits = model(x)
    y_pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()

    # loss
    loss = loss_fn(logits, y)
    # accumulate values
    ls += loss.item()
    acc += accuracy_score(y_true=y.cpu().numpy(), y_pred=y_pred)
    f1 += f1_score(y_true=y.cpu().numpy(), y_pred=y_pred)

    # back propagation
    loss.backward()
    # optmizer step
    optimizer.step()

  # compute averages
  ls /= len(train_dl)
  acc /= len(train_dl)
  f1 /= len(train_dl)

  # return values
  return ls, acc, f1

# function for model testing
def test_batches(model:torch.nn.Module, test_dl:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module, device:str) -> tuple[float, float, float]:
  """Evaluates model on all batches of test-set DataLoader and returns
      average test loss, accuracy and f1_score

  Parameters
  -------
    model: torch.nn.Module
        The model being evaluated

    test_dl: torch.utils.data.DataLoader
        DataLoader for test data

    loss_fn: torch.nn.Module
        Function used to calculate loss

    device: str
        The device on which computation occurs

  Returns
  -------
    ls: float
        average test loss across all batches of data
    acc: float
        average test accuracy across all batches of data
    f1: float
        average test f1_score across all batches of data
  """
  ls, f1, acc = 0, 0, 0

  # evaluation-mode
  model.eval()

  with torch.inference_mode():
    for x, y in test_dl:
      # move x, y to device
      x, y = x.to(device), y.to(device)

      # forward pass
      logits = model(x)
      y_pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()

      # loss
      loss = loss_fn(logits, y)

      # accumulate values
      ls += loss.item()
      acc += accuracy_score(y_true=y.cpu().numpy(), y_pred=y_pred)
      f1 += f1_score(y_true=y.cpu().numpy(), y_pred=y_pred)

  # compute averages
  ls /= len(test_dl)
  acc /= len(test_dl)
  f1 /= len(test_dl)

  # return values
  return ls, acc, f1

# function to return prediction labels (y_pred) and prediction probabilities (y_proba)
def get_preds_proba(model:torch.nn.Module, test_dl:torch.utils.data.DataLoader,
                    device:str) -> tuple[np.ndarray, np.ndarray]:
  """A function that returns y_pred and y_proba from the passed DataLoader

  Parameters
  -------
    model: torch.nn.Module
        A neural network that subclasses torch.nn.Module

    test_dl: torch.utils.data.DataLoader
        A DataLoader for the test dataset

  Returns
  -------
    y_pred: np.ndarray
        A numpy ndarray with prediction labels

    y_proba: np.ndarray
        A numpy ndarray with prediction probabilities
  """
  # empty lists
  y_preds, y_proba = list(), list()
  with torch.inference_mode():
    model.eval() # set eval mode
    for x, _ in test_dl:
      # move x to device
      x = x.to(device)

      # make prediction
      logits = model(x)

      # prediction and probabilites
      proba = F.softmax(logits, dim=1)
      pred = F.softmax(logits, dim=1).argmax(dim=1)

      # append
      y_preds.append(pred)
      y_proba.append(proba)

  y_preds = torch.concatenate(y_preds).cpu().numpy()
  y_proba = torch.concatenate(y_proba).cpu().numpy()

  return y_preds, y_proba
