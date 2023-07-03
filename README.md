# Facial Emotion Recognition

This is an extension of my CS172B project from the University of California, Irvine (Previous work: [YutongLei2020/CS-172B-Project on GitHub](https://github.com/YutongLei2020/CS-172B-Project)). 

The project focuses on building and training a custom CNN model for human facial emotion recognition and utilizing the model for real-time emotion predictions. Due to the presence of numerous labeling errors in the FER2013 dataset, along with some images that do not even represent human facial expressions, I decided to utilize the FER2013+ dataset for annotation. This dataset, relabeled by Microsoft ([microsoft/FERPlus on GitHub](https://github.com/microsoft/FERPlus)), provides improved label annotations for the Emotion FER dataset. The CNN model is a custom Resnet18 architecture built from scratch using Keras. Additionally, I have implemented random erasing data augmentation techniques to enhance the model's performance.

### Datasets
Fer2013：https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Fer+：https://github.com/Microsoft/FERPlus

### Usage
Please download both CSV files before running the following commands.
To train the Resnet 18 model, run the following:
```
python train.py
```
To evaluate the model, run the following:
```
python evaluate.py
```
For the live camera emotion prediction, run the following:
```
python live_cam_pred.py
```
### Evaluation

### References: 
1.	He, Kaiming, et al. ‘Deep Residual Learning for Image Recognition’. ArXiv [Cs.CV], 2015, http://arxiv.org/abs/1512.03385. arXiv.
2.	Zhong, Zhun, et al. ‘Random Erasing Data Augmentation’. ArXiv [Cs.CV], 2017, http://arxiv.org/abs/1708.04896. arXiv
3.	Barsoum, Emad, et al. ‘Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution’. CoRR, vol. abs/1608.01041, 2016, http://arxiv.org/abs/1608.01041.
