# Assignment 2: Recurrent Neural Networks

The second assignment will cover the topic of Recurrent Neural Networks, including backpropagation over time and LSTMs. All details can be found in the PDF document **assignment_2.pdf**.

Unlike the first assignment, there are no unittests this time. We will use PyTorch and its autograd function throughout the assignment.

### Prerequisites

You can use the same environment that you used for the first assignment. 

The first task can be mostly performed on your own computer (CPU), but especially for the second task, you will require a GPU to speed up your training. Hence, we suggest to run experiments on SURFSARA. 

## Task 1. RNNs and LSTMs

For the first task, you will compare Long-Short Term Networks (LSTM) with a specific LSTM variant. You have to implement both network modules in the dedicated files from scratch (i.e. you are not allowed to use `nn.LSTM`, but other functionalities from PyTorch like `nn.Linear`). The datasets are provided in `datasets.py` and can be used without any changes. 

The file `train.py` gives a initial structure for training your models. Make sure to integrate all (hyper-)parameters that are given for the `ArgumentParser`. Feel free to add more parameters if needed.

## Task 2. Text Generation

In the second task, you will use the built-in LSTM function, `nn.LSTM`, to generate text.

### Training Data

Make sure you download the books as plain text (.txt) file. Possible sources to get books from are:

1. Project Gutenberg, Dutch: https://www.gutenberg.org/browse/languages/nl
2. Project Gutenberg, English: https://www.gutenberg.org/browse/languages/en

Feel free to use other languages and/or other sources. Remember to include the datasets/books you used for training in your submission.

### Bonus questions

The assignment contains one bonus questions in which you can experiment further with your model. Note that they are not strictly necessary to get full points on this assignment (max. 100 points), and might require more effort compared to other questions for the same amount of points.

## Task 3. Graph Neural Networks

In the third task, you will have to answer a pen-and-paper questions about Graph Neural Networks. No implementation will be needed. Sources for explaining GNNs can be found in the assignment.

## Report

Similar to the first assignment, we expect you to write a small report on the study of recurrent neural networks, in which you answer all the questions in the assignment. Please, make your report to be self-contained. The page limit of the report is 12 pages.

### Deliverables

Create zip archive with the following structure (note that you only need to have the implementation for your assigned LSTM variant):

```
lastname_assignment_2.zip
│   report_lastname.pdf
│   part_1/
│      datasets.py
│      stm.py
|      gru.py
|      bi_lstm.py
|      peep_lstm.py
│      train.py
│ 
│   part_2/
│      dataset.py
│      model.py
│      train.py
```

Replace `lastname` with your last name. In case you have created any other python files to obtain results in the report, include those in the corresponding directory as well.

Submit your ZIP file on Canvas --> Assignments --> Hand-in Assignment 2.

The deadline for the assignment is the **29th November, 23:59**.

