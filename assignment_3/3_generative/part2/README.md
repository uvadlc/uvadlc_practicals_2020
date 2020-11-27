# Assignment 3, Part 2: Generative Adversarial Networks

This folder contains the template code for implementing your own GAN model. This corresponds to Question 2.3 and 2.4 in the assignment. We will train the model on generating MNIST images. The code is structured in the following way:

* `mnist.py`: Contains a function for preparing the dataset and providing a data loader for training.
* `models.py`: Contains template classes for the Generator and Discriminator
* `train_torch.py`/`train_pl.py`: Contains training functionalities such as the training loop, logging, saving, etc. You can choose between using PyTorch Lightning (`train_pl.py`) or plain PyTorch (`train_torch.py`), and **only need to implement one of them**. We have provided you with logging utilities and general code structure so that you can focus on the important parts of the GAN model.
* `unittests.py`: Contains unittests for the Generator and Discriminator network. It will hopefully help you debugging your code. Your final code should pass these unittests.
* `utils.py`: Contains logging utilities for TensorBoard. Only needed for `train_torch.py`.

A lot of code is already provided to you. Try to familiarize yourself with the code structure before starting your implementation. 
Your task is to fill in the missing code pieces (indicated by `NotImplementedError` or warnings printed). The main missing pieces are:

* In `model.py`, you need to implement the Generator and Discriminator network. We suggest a network architecture in the comments.
* The training files `train_torch.py`/`train_pl.py` are build up in a similar manner. In both, you need to implement:
  * The optimizer definition. As we train two models with different loss functions, you should use two separate optimizers here. One optimizer handles the parameters of the Discriminator, and the second handles the parameters of the Generator. 
  * A `generator_step` function which returns the loss for a single training iteration of the Generator.
  * A `discriminator_step` function which returns the loss for a single training iteration of the Discriminator.
  * A `sample` function that creates new images with the Generator. You shoud log/save those in the function `generate_and_save`/callback `GenerateCallback`
  * A `interpolation` function that randomly samples pairs of latent vectors between which we interpolate and look at the generated images. You should log/save those images in the function `interpolate_and_save`/callback `InterpolationCallback`
  * Additionally in the `train_torch.py` file, you need to implement a training loop for a single epoch in `train_gan`. PyTorch Lightning automatically does this using internal code.
  
Default hyperparameters are provided in the `ArgumentParser` object of the respective training functions. Feel free to play around with those to familiarize yourself with the effect of different hyperparameters. Nevertheless, your model should be able to generate decent images with the default hyperparameters.
  If you test the code on your local machine, you can use the argument `--progress_bar` to show a training progressbar. Remember to not use this on Lisa as it otherwise fills up your SLURM output file very quickly. It is recommended to look at the TensorBoard there instead.
  The training time with the default hyperparameters is less than 20 minutes on a NVIDIA GTX1080Ti (GPU provided on Lisa).

Note that in the provided template, we separate the Generator and Discriminator step into two independent functions. Another way of implementing it would be to combine them in a single function, and re-use the generated images by the Generator for the loss calculation of the Discriminator. However, this approach is more error prone and requires to take care of the computation graph. For simplicity, we separated the two steps in the implementation here. The speed-up of the other implementation variant is neglectable for our usecase given the small size of the Generator.
