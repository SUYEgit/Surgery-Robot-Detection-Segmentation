# Mask R-CNN for Object Detection and Segmentation

This is a project for surgery robot target detection and segmentation based on implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) by (https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. 
![Instance Segmentation Sample](assets/center.png)
The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Instruction and training code for the surgery robot dataset
* Pre-trained weights on MS COCO and ImageNet
* Jupyter notebooks to visualize the detection result
* Example of training on your own dataset, with emphasize on how to build and adapt codes to dataset with multiple classes.

# Instance Segmentation Samples on Robot Dataset
The model is trained based on pre-trained weights for MS COCO. 
![Instance Segmentation Sample2](assets/left.png)
![Instance Segmentation Sample2](assets/right.png)

# Video Demo


# Training on Your own Dataset
Pre-trained weights from MS COCO and ImageNet are provided for you to fine-tune over new dataset. Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.

In summary, to train the model you need to modify two classes in ```surgery.py```:
1. ```SurgeryConfig``` This class contains the default configurations. Modify the attributes for your training, most importantly the ```NUM_CLASSES```.
2. ```SurgeryDataset``` This class inherits from ```utils.Dataset``` which provides capability to train on new dataset without modifying the model. In this project I will demonstrate with a dataset labeled by VGG Image Annotation(VIA). If you are also trying to label a dataset for your own images, start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). 
First of all, for training you need to add class in function ```load_VIA```
```
self.add_class("SourceName", ClassID, "ClassName")
#For example:
self.add_class("surgery", 1, "arm")  #means add a class named "arm" with class_id "1" from source "surgery"
......
```
Then extend function ```load_mask``` for reading different class names from annotations
For example, if you assign name "a" to class "arm" when you are labelling, according to its class_id defined in ```load_VIA```
```
class_ids = np.zeros([len(info["polygons"])])
for i, p in enumerate(class_names):
   if p['name'] == 'a':
      class_ids[i] = 1
      ......
```
3. Now you should be able to start training on your own dataset! Training parapeters are mainly included in function ```train``` in ```surgery.py```.

# Prediction and Visualization
3. For visualization: ```detect_and_color_splash``` add class_names
4. dataset:  surgery: train val predict
```
#Train a new model starting from pre-trained COCO weights
python surgery.py train --dataset=/home/.../mask_rcnn/data/surgery/ --weights=coco  

#Train a new model starting from pre-trained ImageNet weights
python surgery.py train --dataset=/home/.../mask_rcnn/data/surgery/ --weights=imagenet

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python surgery.py train --dataset=/home/.../mask_rcnn/data/surgery/ --weights=last
```

```
#Detect and color splash on a single image with the last model you trained.
#This will find the last trained weights in the model directory.
python surgery.py splash --weights=last --image=/home/...../*.jpg

#Detect and color splash on images in a directory
python surgery.py detect --weights=last --dataset=/home/.../mask_rcnn/data/surgery/ --subset=predict

#Detect and color splash on a video with a specific pre-trained weights of yours.
python sugery.py splash --weights=/home/.../logs/mask_rcnn_surgery_0030.h5  --video=/home/simon/Videos/Center.wmv
```


The training schedule, learning rate, and other parameters should be set in the `train` function in `surgery.py`.


# Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

See examples in `samples/shapes/train_shapes.ipynb`, `samples/coco/coco.py`, `samples/balloon/balloon.py`, and `samples/nucleus/nucleus.py`.


## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.


## Installation
1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

# Projects Using this Model
If you extend this model to other datasets or build projects that use it, we'd love to hear from you.

### [4K Video Demo](https://www.youtube.com/watch?v=OOT3UIXZztE) by Karol Majek.
[![Mask RCNN on 4K Video](assets/4k_video.gif)](https://www.youtube.com/watch?v=OOT3UIXZztE)

### [Images to OSM](https://github.com/jremillard/images-to-osm): Improve OpenStreetMap by adding baseball, soccer, tennis, football, and basketball fields.

![Identify sport fields in satellite images](assets/images_to_osm.png)

### [Splash of Color](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). A blog post explaining how to train this model from scratch and use it to implement a color splash effect.
![Balloon Color Splash](assets/balloon_color_splash.gif)


### [Segmenting Nuclei in Microscopy Images](samples/nucleus). Built for the [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)
Code is in the `samples/nucleus` directory.

![Nucleus Segmentation](assets/nucleus_segmentation.png)

### [Mapping Challenge](https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn): Convert satellite imagery to maps for use by humanitarian organisations.
![Mapping Challenge](assets/mapping_challenge.png)


