# Project 1

Due: May 1, 2025  
姓名: 雍崔扬  
学号: 21307140051

## 1. Problem Setting

In this problem we will investigate handwritten digit classification.   
MNIST (Modified National Institute of Standards and Technology database)   
is a large database of handwritten digits commonly used for training various image processing systems.   
The database is also widely used for training and testing in the field of machine learning.   
It was created by "re-mixing" the samples from NIST's original datasets.   
The dataset contains $60,000$ training images and $10,000$ testing images.   
Each image is a $28\times 28$ pixel grayscale image and is labeled with the correct digit ($0\sim9$) it represents.   
You need to implement one or more neural network to recognize the handwritten digit,   
and conduct experiment to test your model and conclude the ability of your model.   
After that, you may implement several modifications to your model and test whether the model acts better.   
Do not worry about whether your mode's performance is better than others',   
the effort you’ve made to improve your model matters.   

Please refer to following instruction of writing the report:

- ① The goal of your write-up is to document the experiments you’ve done and your main findings.   
  So be sure to explain the results.   
  Hand in a single PDF file of your report.   
  Enclose a Github link to your codes in your submitted file.   
  You should also provide a link to your dataset and your trained model weights in your report.   
  You may upload the dataset and model into Google Drive or other Netdisk service platform.   
  Also put the name and Student ID in your paper.   
  Lack of code link or model weights link will lead to a penalization of scores.  
- ② You may use Mindspore to do some visualization   
  or do some experiments if your implemented version runs too slow.  
  But you must implement your own version first.

- ③ Note that the goal of this project is to let the students do the practice,   
  and write some basic components of a neural network or CNN.   
  So do not invoke the deep learning functions/modules   
  that can directly give the results of the following questions (e,g, the PyTorch package).   
  The project is very easy, and you do not really need GPUs;   
  just running it on CPU will be enough.



## 2. Questions

Below are additional features you could try to incorporate into your neural network   
to improve performance (the options are approximately in order of increasing difficulty).   
The modifications you make in trying to improve performance are up to you   
and you can even try things that are not on the question list.   
But, let's stick with neural networks models and only use one neural network (no ensembles).   
Remember to write in your reports about what modifications you’ve made and the effectiveness of the modification.



