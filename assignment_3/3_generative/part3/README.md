# Assignment3 - Part 3 - Toy application: Transforming Gaussian densities with flows

#### Corresponds to Question 3.4 in Assignment 3.
This folder contains the template code for implementing parts of a generative
flow model to transform a simple Gaussian into a bimodal Gaussian density. We
will train the model on data generated from the target density. The code is
structured in the following way:

* `distributions.py`: Contains classes for bimodal and unimodal multivariate
                      Gaussian densities with diagonal covariance matrix. It
                      also contains a functionality to broadcast tensor values
                      along dimensions to allow for element-wise multiplication. This will
                      come in handy for coding the log probability function.
* `train.py`: Contains training loop along with the tensorboard experiment
              logger, image saving etc. for your convenience.
* `model.py`: Contains template for Normalizing Flow and Coupling Layer classes.        
* `utils.py`: Contains visualization functions and mesh grid generators.

A lot of code is already provided to you. Try to familiarize yourself with the
code structure before starting your implementation. Your task is to fill in the
missing code pieces (indicated by `NotImplementedError` or warnings printed).
The main missing pieces are:

* First, in `distributions.py`, you need to implement the Unimodal and Bimodal
Gaussian densities. You can make use of the broadcasting functionality we
provided. Per density class you need to implement:
    * A `log_prob` function that should compute the log probabilty of an
      input value.
    * A `sample` function that samples from the models using the
      reparametrization trick.

* The model is implemented in `model.py` file, you need to implement the
  forward and backward pass.
* The training file `train.py` you need to implement a training loop for
  a single epoch.


Default hyperparameters are provided in the `ArgumentParser` object of the
  training functions. Feel free to play around with those to familiarize
  yourself with the effect of different hyperparameters. Nevertheless, your
  model should be able to generate decent images with the default
  hyperparameters. If you test the code on your local machine, you can use the
  argument `--progress_bar` to show a training progress bar. The model can be
  run on CPU. During training you can check the TensorBoard output logs
  to view the current state of the estimated density evaluated on
  points from the input domain.
