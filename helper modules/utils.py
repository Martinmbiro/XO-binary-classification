"""
contains helper functions for tasks like loading & saving models,
earlystopping, plotting training metrics, e.t.c
"""
import torch, pathlib, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import Dataset, Subset
from pathlib import Path
from copy import deepcopy

# declarea class to implement earlystopping
class EarlyStopping:
  """A class that implements EarlyStopping mechanism

  Parameters
  -------
    score_type:
      'metric' for a metric score (eg. f1_score) and 'loss' for
        a loss function (eg. CrossEntropyLoss)

    min_delta: float
      How much of a difference in loss is to be considered worthy to continue training

    patience: int
      The number of epochs to wait after the last improvement before stopping
  """
  def __init__(self, score_type:str, min_delta:float=0.0, patience:int=5): # constructor
    self.counter = 0
    self.patience = patience
    self.min_delta = min_delta
    self.score_type = score_type
    self.best_epoch = None
    self.best_score = None
    self.best_state_dict = None
    self.stop_early = False

    if (self.score_type != 'metric') and (self.score_type != 'loss'):
      err_msg = 'score_type can only be "metric" or "loss"'
      raise Exception(err_msg)

  def __call__(self, model:torch.nn.Module, ep:int, ts_score:float):
    """Pass the following arguments to the object name of EarlyStopping to call this method

    Parameters
    -------
      model: torch.nn.Module
          The object name of the model being trained (subclasses nn.Module)

      ep: int
          The current epoch in the training / optimization loop

      ts_score: float
          The score (loss or metric) being used to decide early stopping mechanism
    """
    if self.best_epoch is None: # for first time:
      self.best_epoch = ep # store current epoch
      self.best_score = ts_score # store current loss as best loss
      # make a copy of current model's state_dict
      self.best_state_dict = deepcopy(model.state_dict())

    # if previous loss - current loss exceeds min_delta: (for loss function)
    elif (self.best_score - ts_score >= self.min_delta) and (self.score_type == 'loss'):
      self.best_epoch = ep # store current epoch
      self.best_score = ts_score # store current loss as best
      # make a copy of current model's state_dict
      self.best_state_dict = deepcopy(model.state_dict())
      self.counter = 0 # restore counter to zero

    # if current metric - previous. metric exceeds min_delta: (for metric)
    elif (ts_score - self.best_score >= self.min_delta) and (self.score_type == 'metric'):
      self.best_epoch = ep # store current epoch
      self.best_score = ts_score # store current loss as best
      # make a copy of current model's state_dict
      self.best_state_dict = deepcopy(model.state_dict())
      self.counter = 0 # restore counter to zero

    else: # otherwise
      self.counter += 1 # increment counter each time
      if self.counter >= self.patience:
        self.stop_early = True

# function to plot train and test results
def plot_train_results(ep_list:list, train_score:list, test_score:list,
                       ylabel:str, title:str, best_epoch:int):
  """A function that plots train and test results against each other

  Parameters
  -------
    ep_list: list
      A list containing all epochs used in the optimization loop

    train_score: list
      A list containing a specific training score from the optimization loop

    test_score: list
      A list containing a specific training score from the optimization loop

    y_label: str
      y-axis label for the plot

    title: str
      Title for the plot

    best_epoch: int
      Best epoch for which early stopping occurred
  """
  f, ax = plt.subplots(figsize=(5, 3), layout='constrained')

  # train loss
  ax.plot(ep_list, train_score, label='Training',
          linewidth=1.7, color='#0047ab')

  # test loss
  ax.plot(ep_list, test_score, label='Validation',
          linewidth=1.7, color='#990000')
  # vertical line (for early stopping)
  if best_epoch is not None:
    ax.axvline(best_epoch, linestyle='--', color='#000000', linewidth=1.0,
             label=f'Best ep ({best_epoch})')

  # axis, title
  ax.set_title(title, weight='black')
  ax.set_ylabel(ylabel)
  ax.set_xlabel('Epoch')
  ax.tick_params(axis='both', labelsize=9)
  plt.grid(color='#e5e4e2')

  # legend
  f.legend(fontsize=9, loc='upper right',
          bbox_to_anchor=(1.28, 0.93),
          fancybox=False)

  plt.show()

# function to plot confusion matrix
def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray):
  """A function that plots Confusion Matrix for all classes

  Parameters
  -------
    y_true: np.ndarray
      An ndarray containing true label values

    y_pred: np.ndarray
      An ndarray containing predicted label values
  """
  # define figure and plot
  _, ax = plt.subplots(figsize=(3.0,3.0), layout='compressed')
  # plot
  ConfusionMatrixDisplay.from_predictions(
      y_true=y_true,
      y_pred=y_pred, cmap='Blues', colorbar=False, ax=ax)

  # set x and y labels
  ax.set_ylabel('True Labels', weight='black')
  ax.set_xlabel('Predicted Labels', weight='black',
                  color='#dc143c')
  # set tick size and position
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  ax.tick_params(axis='both', labelsize=9)

  # change annotation font
  for txt in ax.texts:
    txt.set_fontsize(9)

  plt.show()

# function to save model to specified directory
def save_model(model:torch.nn.Module, path:pathlib.PosixPath):
  """Function to save model to a specified path

  Parameters
  -------
    model: torch.nn.Module
      The model to save

    path: pathlib.PosixPath
      Path to save model's state_dict
  """
  torch.save(obj=model.cpu().state_dict(), f=path)
  print(f"MODEL'S state_dict SAVED TO: {path}")

# function to load model from a specified path
def load_model(model:torch.nn.Module, path:pathlib.PosixPath):
  """Function to load model from a specified path

  Parameters
  -------
    model:torch.nn.Module
        A new object of the model class

    path:pathlib.PosixPath
        Path pointing to a previously saved model's state_dict

  Return
  -------
    model:torch.nn.Module
      model returned after loading state_dict
  """
  # overwrite stat_dict
  model.load_state_dict(
      torch.load(f=path, weights_only=True))

# function to make inference on a single random image
def make_single_inference(model:torch.nn.Module, dataset:torch.utils.data.Dataset,
                          label_map:dict, device:str):
  """Makes inference using a random data point from the test dataset

  Parameters
  -------
    model: torch.nn.Module
      A model (subclassing torch.nn.Module) to make inference

    dataset: torch.utils.data.Dataset
      The Dataset to use for testing purposes

    label_map: dict
      A dictionary maping indices to labels (eg. {0:'O', 1:'X'})

    device: str
      Device on which to perform computation
  """
  # get random image from test_set
  idx = np.random.choice(len(dataset))
  img, lb = dataset[idx]

  # make prediction
  with torch.inference_mode():
    model.to(device) # move model to device
    model.eval() # set eval mode
    lgts = model.to(device)(img.unsqueeze(0).to(device))
    pred = F.softmax(lgts, dim=1).argmax(dim=1)

  # print actual retrieved image
  plt.figure(figsize=(1.0, 1.0))
  # title with label
  if pred==lb:
    plt.title(
        f'Actual: {label_map[lb]}\nPred: {label_map[pred.item()]}',
        fontsize=8)
  else: # if labels do not match, title = with red colour
    plt.title(
        f'Actual: {label_map[lb]}\nPred: {label_map[pred.item()]}',
        fontsize=8, color='#de3163', weight='black')
  plt.axis(False)
  plt.imshow(img.squeeze(), cmap='gray')
  plt.show()

# function to make inference on multiple random images
def make_multiple_inference(model:torch.nn.Module, dataset:torch.utils.data.Dataset,
                            label_map:dict, device:str):
  """Makes inference using a random data point from the test dataset

  Parameters
  -------
    model: torch.nn.Module
      A model (subclassing torch.nn.Module) to make inference

    dataset: torch.utils.data.Dataset
      The Dataset used for evaluation purposes

    label_map: dict
      A dictionary maping indices to labels (eg. {0:'O', 1:'X'})

    device: str
      Device on which to perform computation
  """
  # get array of 12 random indices of images in test_dataset
  indices = np.random.choice(len(dataset),
                                  size= 12, replace=False)
  # create subset from the 12 indices
  sub_set = Subset(dataset=dataset, indices=indices)

  # define a figure and subplots
  f, axs = plt.subplots(2, 6, figsize=(6,5), layout='compressed')

  # move model to device & set eval mode
  model.to(device)
  model.eval()

  # loop through each subplot
  for i, ax in enumerate(axs.flat):
    img, lb = sub_set[i] # return image and label

    # make inference on image retuned
    with torch.inference_mode():
      lg = model(img.unsqueeze(0).to(device))
      pred = F.softmax(lg, dim=1).argmax(dim=1)

    ax.imshow(img.squeeze(), cmap='gray')
    ax.axis(False)
    if pred==lb:
      ax.set_title(
          f'Actual: {label_map[lb]}\nPred: {label_map[pred.item()]}',
          fontsize=8)
    else: # if labels do not match, title = with red colour
      ax.set_title(
          f'Actual: {label_map[lb]}\nPred: {label_map[pred.item()]}',
          fontsize=8, color='#de3163', weight='black')

  f.suptitle('Inference Made on 12 Random Test Images',
            weight='black',
            y=0.83)
  plt.show()
