# Alcohol Consumption Detection from Periocular NIR Images Using Capsule Network
Juan Tapia, Enrique Lopez Droguett and Christoph Busch

Paper link: https://ieeexplore.ieee.org/document/9956573

# Description
This research proposes a method to detect alcohol consumption from a Near-Infra-Red (NIR) periocular eye images. The study focuses on determining the effect of external factors such as alcohol on the Central Nervous System (CNS). The goal is to analyse how this impacts on iris and pupil movements and if it is possible to capture these changes with a standard iris NIR camera. This paper proposes a novel Fused Capsule Network (F-CapsNet) to classify iris NIR images taken under alcohol consumption subjects. The results show the F-CapsNet algorithm can detect alcohol consumption in iris NIR images with an accuracy of 92.3% using half of parameters than the standard Capsule Network algorithm. This work is a step forward for developing an automatic system to estimate "Fitness for Duty" and prevent accidents due to alcohol consumption.

# Database

A total of 600 images of volunteers not under the influence of alcohol were captured and 2,400 images were taken after each volunteer had ingested 200 ml of alcohol (Images taken in intervals of 15 minutes after consumption). The database was divided into 70\% and 30\% for Training and Testing. The partition is a subject-disjoint database.

<p align="center">
<img width="357" alt="Iris_example" src="https://user-images.githubusercontent.com/45126159/172894528-9f4d44b9-4d2d-4c9c-9f71-66dce00940bb.png">
</p>




# Database link

Question about the dataset to: juan.tapia-farias@h-da.de or jtapiafarias@ing.uchile.cl.

-Only for research purpose and institucional emails belongs to advisor or professors (University).

Downloading -IEEE dataPort: https://ieee-dataport.org/documents/nir-iris-images-under-alcohol-effect

A health committee team evaluated the captured process before it started. A consent form was used to capture images for all the volunteers as requested by the health committee.

# Description.

- 5 sessions of image capturing were done for each person at intervals of 00, 15, 30, 45 and 60 minutes after having ingested alcohol.
- There are different people for each folder "Grupo_X". Each one have a folder with the name of the sensors used in the capturing session (LG/Iritech). Each sensor folder have 5 inner folders for each interval (in minutes) in the capturing session.

# Annotated information.

As an additional information, the coordinates of iris, pupil and sclera were included to be used in segmentation.

- Each folder of each session have a json file which have manual annotations of every region of interest of the eye (pupil, iris, sclera)
- [VIA v2.0.5](https://www.robots.ox.ac.uk/~vgg/software/via/) was used to label each ROI of the eyes.
- Each ROI of each image was labeled with distinct figures disposed in VIA program (polygon, ellipse, circle) specified in json files.
- Polygon figure have a coords set like (x, y) points.
- Circle figure have the center denoted as (x, y) coord and his radius (r).
- Ellipse figure have the center denoted as (x, y) coord, the minor and the major radii.
- File "eye_attributes.json" correspond to region/file attribute of VIA program, needed to identify each ROI.

![image](https://github.com/jedota/Iris_alcohol_classification/blob/main/static/seg_image.png?raw=true)

### From image name "E_5_7_0_L_F_N_N_1994_3_2017.bmp" we can extract the following features:

- The number 5 indicates that it belongs to the "Grupo_5" folder.
- The number 7 indicates that it belongs to the subject 7 of the corresponding group.
- The number 0 indicates that it belongs to the timeset 0. This number can be (0, 1, 2, 3 or 4) depending on the capturing interval (00, 15, 30, 45 or 60 minutes)
- The L letter means that is a "Left" eye.
- The F letter means that is a Female subject.
- 1994 means the date of birth.
- The number after date of birth is the corresponding frame of the video.
- 2017 correspond to the capturing date.


# Set split
- Train, test and validation set are disposed in text files into [set_split](https://github.com/Choapinus/alcohol-db/tree/master/set_split) folder

</p>


# Citation
If you use MAAL dataset, please cite the following paper:
```
@INPROCEEDINGS{9956573,
  author={Tapia, Juan and Droguett, Enrique Lopez and Busch, Christoph},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Alcohol Consumption Detection from Periocular NIR Images Using Capsule Network}, 
  year={2022},
  volume={},
  number={},
  pages={959-966},
  doi={10.1109/ICPR56361.2022.9956573}}
```


# License

The dataset, the implementation, or trained models, use is restricted to research purpuses. The use of the dataset or the implementation/trained models for product development or product is not allowed. This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
