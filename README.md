# Fake_face_detection
The rapidly improving image generation technology had drastically increases the difficulty of human detection of whether or not the images are real. And although there are already many GAN image detectors online, none of them has showed their generalization capability to unseen samples. Hence, this project experiment with different combinations of the best model architectures and preprocessing methods and test them with image from more advanced GAN or manipulated image to evaluate the robustness of these models and the performance of different preprocessing methods.
![SEI_149850353](https://github.com/HertZMX/Fake_face_detection/assets/107277409/d1fe774d-3a54-47af-9c2b-a437ba0e4f32)

## Repository and Code Structure

This github repository is mainly compose of two part: one for model training and one for AutoML on Google Cloud: 
- The models folders contains four different model created for this project and how to run them. As well as another ipynb file that stores the structures of all four models. 
- The autoML folders contains a python file that includes code which allows you create all the file you need in order to create an autoML on google. You need a input_file.csv to create a dataset an!
d you need the jsonl file to do batch prediction

The other file contains code that allow you to preprocess the data, apply preprocess method to images. And lastly there are codes we used to evaluate the performance of our model

---
The project consists of three main segments. 

The initial segment is dedicated to data preparation. It involves utilizing pre-defined data preprocessing functions, housed within a .py file. These functions allow for the generation of preprocessed images, which are then saved in a specified directory. 

The second segment is about model training. This segment includes five models, each detailed in a .ipynb file. These files follow a similar structure: initially, a data loader function is utilized, followed by the loading of the training and testing datasets. After this, image examples from each class are visualized, the model is defined, and then trained and evaluated. The final step is generating output for the predictions of the test datasets, allowing for later comparison. 

The third segment of the project is evaluation. This involves collating all the test predictions in one file and assessing them using another evaluation file. The evaluation file loads the relevant CSV data, generates plots for each model, and produces plots for subgroup results.

To execute the code of the model segment, download the .ipynb file and run it from start to end. All the parts involving the directory should be adjusted to match the location of datasets.

## Conclusion

We discovered that the models of ResNet18, ShallowNetV3, Xception, and DenseNet121 all had accuracy below 60%. In contrast, AutoML achieved a performance rate of around 77%, and ResNet50 with pretrain had an accuracy rate of approximately 73%. In all four models and the AutoML model, the data preprocessing technique did not seem to improve performance with this dataset, likely due to the low resolution of the images. Further experiments may consider using higher resolution images. 

![download](https://github.com/HertZMX/Fake_face_detection/assets/107277409/6a86bf3c-c1af-4f23-8d37-6640fa21aab9)

Subgroup comparisons indicated that certain subgroups could significantly influence results, with the accuracy of StyleGAN2 reaching almost 100% in AutoML model and ResNet50 model and highest performance on all other four models.

<br />

<img src="https://github.com/HertZMX/Fake_face_detection/assets/107277409/99d6a2f2-476c-4fda-b7c9-5e8753c951a1" alt="Image 1" width="300px" />
<img src="https://github.com/HertZMX/Fake_face_detection/assets/107277409/85bbc13d-2716-43a2-8881-604c9095ed28" alt="Image 2" width="300px" />
<img src="https://github.com/HertZMX/Fake_face_detection/assets/107277409/758f4f3b-c71a-4d1e-b90f-610fd6a820e1" alt="Image 3" width="300px" />

<br />
                                                                                                                                     
In terms of robustness, despite AutoML's superior accuracy, the performance is poor when testing with unseen datasets, such as StyleGAN3 and Photoshop. In contrast, the other five models demonstrated greater robustness, with pretrained ResNet50 showing the highest robustness, achieving an accuracy of 70%. 

![image](https://github.com/HertZMX/Fake_face_detection/assets/95679749/a8c8d6f0-029e-4d92-91e8-105e2a497bb2)

Another pattern is shown in all of the models we experimented. Although ResNet50 with pretrain improved the best testing accuracy, but the testing accuracy is around the same from the first to 300th epoch. The training accuracy can be as high as 90%, indicating we are not underfitting, and the training accuracy increase during the processes also indicates we are not overfitting. This phenomenon should be further investigated and experimented upon. 

In conclusion, our chosen models performed with lower accuracy than the AutoML model but demonstrated higher robustness. The performance can be improved by adding pretraining. The benefit of preprocessing did not manifest in this dataset, but it could be experimented more in future studies.
