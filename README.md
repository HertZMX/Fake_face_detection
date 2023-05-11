# Fake_face_detection
The rapidly improving image generation technology had drastically increases the difficulty of human detection of whether or not the images are real. And although there are already many GAN image detectors online, none of them has showed their generalization capability to unseen samples. Hence, this project experiment with different combinations of the best model architectures and preprocessing methods and test them with image from more advanced GAN or manipulated image to evaluate the robustness of these models and the performance of different preprocessing methods.
![FsEy5bVWcAAs4xN](https://github.com/HertZMX/Fake_face_detection/assets/107277409/b41bf052-5968-4695-9d22-504364919242)

## Repository and Code Structure

This github repository is mainly compose of two part: one for model training and one for AutoML on Google Cloud: 
- The models folders contains four different model created for this project and how to run them. As well as another ipynb file that stores the structures of all four models. 
- The autoML folders contains a python file that includes code which allows you create all the file you need in order to create an autoML on google. You need a input_file.csv to create a dataset and you need the jsonl file to do batch prediction

The other file contains code that allow you to preprocess the data, apply preprocess method to images. And lastly there are codes we used to evaluate the performance of our model
