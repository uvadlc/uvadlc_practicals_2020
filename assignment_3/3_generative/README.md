# Assignment 3: Deep Generative Models

You can find all necessary instructions in **assignment_3.pdf**. The assignment consists of a combination of pen-and-paper/theory exercises and implementation questions. The details for the implementation can be found in the assignment and the individual README files in the respective code folders.

## Submission
When you submit your code, make sure to use the same structure as we provide in this assignment. If you add additional files, clearly mention this in your report and explain what code can be found in there. **Do not include the data of the MNIST dataset in your submission, and also not any logs of your results.** This will reduce the size of the submission and ensure an easy upload to Canvas.

## Updates
- 2 Dec: Updated typos in Assignment 3 PDF <br />
        - Section 1.3: Kullback-Leibner --> Kullback-Leibler <br />
        - Section 1.5: Mapping of Sigma and Mu below equation 16. Has to be from M --> D. <br />
- 8 Dec: Updated docstring Assignment 3 distributions.py.<br />
- 21 Dec:
        - Part 1: Number of channels for an MNIST image should be 1. part1/cnn_encoder_decoder.py at line#64 <br/>
        - Part 2: No means, only samples from GANs: part2/train_pl.py at line#179 <br/>
        - Part 3: Fix extent argument in utils.py plot_contours <br/>
        - Part 3: Inverse is for z to x, not x to z: part3/model.py at line#57 <br/>


  
