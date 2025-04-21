# XO Binary Classification

<p align="center">
  <img src='pics/modular.png'  width='500'/>
</p>

Hello again 👋
+ [Modular programming](https://en.wikipedia.org/wiki/Modular_programming) is a software design technique where the code is broken down into smaller independent pieces called _modules_ that handle specific functions. This promotes reusability, maintainability, and cleaner code.
+ In this repository, I solve an end-to-end binary image classification problem on `X` & `O` image classes from the [`EMNIST`](https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html) dataset, while implementing modular programming. The training dataset consists of `9,600` images, while the test dataset consists of `1,600` images - all grayscale, with `28` pixels in height and width.
+ The model architecture used for this exercise is a tweaked version of the [TinyVGG model](https://www.youtube.com/watch?v=HnWIHWFbuUQ), that I'd built in a previous repository - [Pytorch computer vision basics](https://github.com/Martinmbiro/Pytorch-computer-vision-basics)
+ The first notebook, `01. XO modular.ipynb` entails turning reusable code into modules, while the second notebook, `02. XO end to end.ipynb` is where I put everything together.
+ Comments, working code, and links to the latest official documentation are included every step of the way. There's links to open each notebook (_labeled 01...02_) in Google Colab - feel free to play around with the code.

## Milestones 🏁
**Concepts covered in this exercise include:**  
1. [x] Data wrangling and visualization
2. [x] Training and evaluating a binary image classification model - build using [`PyTorch`](https://pytorch.org/)
3. [x] Regularization using [Early stopping](https://www.linkedin.com/advice/1/what-benefits-drawbacks-early-stopping#:~:text=Early%20stopping%20is%20a%20form,to%20increase%20or%20stops%20improving.), [`nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d), [`nn.BatchNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d), and [`nn.Dropout`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)
4. [x] Modular programming

## Tools ⚒️
1. [`Google Colab`](https://colab.google/) - A hosted _Jupyter Notebook_ service by Google.
2. [`PyTorch`](https://pytorch.org/) -  An open-source machine learning (ML) framework based on the Python programming language that is used for building **Deep Learning models**
3. [`scikit-learn`](https://scikit-learn.org/stable/#) - A free, open-source library that offers Machine Learning tools for the Python programming language
4. [`numpy`](https://numpy.org/) - The fundamental package for scientific computing with Python
5. [`matplotlib`](https://matplotlib.org/) - A comprehensive library for making static, animated, and interactive visualizations in Python
6. [`torchinfo`](https://github.com/TylerYep/torchinfo) - A library for viewing model summaries in PyTorch

## Results 📈
> On a scale of `0` -> `1`, the final best-performing model achieved:
+ A weighted `precision`, `recall`, and `f1_score` of `1.00`
+ An overall model accuracy of `0.9956`
+ An overall `roc_auc_score` of `1.00`

> The saved model's `state_dict` can be found in the drive folder linked [here](https://drive.google.com/drive/folders/1gWNM3EXADR0__wuUyzAipIg7vu47aEh3?usp=drive_link)


## Reference 📚
+ Thanks to the insight gained from [`Daniel Bourke`](https://x.com/mrdbourke?s=21&t=1Fg4dWHIo5p7EaMHhv2rng) and [`Modern Computer Vision with Pytorch, 2nd Edition`](https://www.packtpub.com/en-us/product/modern-computer-vision-with-pytorch-9781803240930)
+ Not forgetting these gorgeous gorgeous [`emojis`](https://gist.github.com/FlyteWizard/468c0a0a6c854ed5780a32deb73d457f) 😻

> _Illustration by [`Storyset`](https://storyset.com)_ ♥

