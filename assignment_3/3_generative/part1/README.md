# Assignment 3, Part 1: Variational Autoencoders

This folder contains the template code for implementing your own VAE model. This corresponds to Questions 1.9 to 1.11 in the assignment. This also includes the bonus question 1.12. 
We will train the model on generating binary MNIST images. The original MNIST dataset contains images with pixel values between 0 and 1. To binarize those, we set those pixels to 1 that have a value greater than 0.5, and all others to 0. 

The code is structured in the following way:
* `bmnist.py`: Contains a function for preparing the binarized dataset and providing a data loader for training, validation and testing.
* `mlp_encoder_decoder.py`: Contains template classes for the Encoder and Decoder based on an MLP
* `cnn_encoder_decoder.py`: Contains template classes for the Encoder and Decoder based on an CNN (only needed for bonus question 1.12)
* `train_torch.py`/`train_pl.py`: Contains training functionalities such as the training loop, logging, saving, etc. You can choose between using PyTorch Lightning (`train_pl.py`) or plain PyTorch (`train_torch.py`), and **only need to implement one of them**. We have provided you with logging utilities and general code structure so that you can focus on the important parts of the VAE model.
* `utils.py`: Contains functionalities that are required for training, such as the reparameterization trick, the KL divergence, bpd calculation and manifold generation.
* `unittests.py`: Contains unittests for the Encoder and Decoder networks, as well as functions of `utils.py`. It will hopefully help you debugging your code. Your final code should pass these unittests.

A lot of code is already provided to you. Try to familiarize yourself with the code structure before starting your implementation. 
Your task is to fill in the missing code pieces (indicated by `NotImplementedError` or warnings printed). The main missing pieces are:
* In `mlp_encoder_decoder.py`, you need to implement the Encoder and Decoder network as MLP. We suggest a network architecture in the comments.
* In `utils.py`, you need to implement couple of smaller functions that are used in the training file:
  * A `sample_reparameterize` function that should implement the reparameterization trick, i.e. sampling from a distribution with means and standard deviation that require gradients
  * A `KLD` function that should implement the KL divergence of the unit Gaussian prior and the predicted distribution
  * A `elbo_to_bpd` function that should implement the bits-per-dimension metric. The ELBO represents the negative log likelihood given by the ELBO objective of the VAE (reconstruction loss plus regularization loss)
  * A `visualize_manifold` function that should implement the visualization of the manifold in latent space. See question 1.11 for details.
* The training files `train_torch.py`/`train_pl.py` are build up in a similar manner. In both, you need to implement:
  * A `forward` function which returns the reconstruction and regularization loss, as well as the bits per dimension metric for a single batch.
  * A `sample` function that creates new images with the Decoder. You shoud log/save those in the function `sample_and_save`/callback `GenerateCallback`
  * Additionally in the `train_torch.py` file, you need to implement a training and test loop for a single epoch in `train_vae` and `test_vae`. PyTorch Lightning automatically does this using internal code.
  
Default hyperparameters are provided in the `ArgumentParser` object of the respective training functions. Feel free to play around with those to familiarize yourself with the effect of different hyperparameters. Nevertheless, your model should be able to generate decent images with the default hyperparameters.
  If you test the code on your local machine, you can use the argument `--progress_bar` to show a training progressbar. Remember to not use this on Lisa as it otherwise fills up your SLURM output file very quickly. It is recommended to look at the TensorBoard there instead.
  The training time with the default hyperparameters is less than 20 minutes on a NVIDIA GTX1080Ti (GPU provided on Lisa).
