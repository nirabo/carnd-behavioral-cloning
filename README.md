# Behavioral Cloning - Udacity CarND Project 3

[//]: # (Image References)

[initial]: ./docs/initial.png "Initial distribution prior augmentation"
[track-1-left]: ./docs/tr1_lcr.png "Track one left"
[track-1-left-flipped]: ./docs/tr1_lcr_flip.png "Track one left"
[track-1-right]: ./docs/tr1_lcr_right.png "Track one right"
[track-1-right-flipped]: ./docs/tr1_lcr_right_flip.png "Track one right"
[combined]: ./docs/combined.png "Combined all tracks"
[combined-reduced-centre]: ./docs/combined_reduce_centre.png "Combined all tracks with reduced centre images"
[with-and-without-noise]: ./docs/with_without_noise.png "With and without noise"
[with-modified-brightness]: ./docs/with_modified_brightness.png "With modified brightness"
[track-2]: ./docs/tr2_lcr.png "Track two"
[model]: ./docs/model.png
## Summary
In this excercise, solely the visual input from three front-facing cameras in a
simulator are used
to predict the steering angle of the car-simulator. The model was trained on two
tracks with varios direction changes and lane changes. Several data-augmentation
techniques and CNN architectures where tested and applied in order to improve upon the training process. The resulting model performs well on both tracks, at 17mph
on the mountain track and 25mph on the lake track.

## Introduction

End-to-End (E2E) learning  through convolutional deep neural networks was shown
recently to perform better then hand-crafted feature extraction, object detection,
behavior inference and decision making in the autonomous vehicle domain [1]. This
is attributed to the ability of deep CNNs to innately capture complex details of
the scene and their relations. The resulting proposal is that the stream of a monocular
vision sensor will be enough to make the car drive itself around complex terrains.

### Data Sampling
In this excercise a video stream of a car being driven in simulation around two
tracks was recorded, a 'lake' track and a 'mountain' track, named track-1 and track-2
respectively.



***Track-1*** is wide with interrupted lane markings, circumventing a lake.
Although being at least twice as wide as the car in the simulator, the track proved
to be difficult to be learned due to varios ligthning condition changes, texture
changes on the road and surroundings, bridge crossings and interrupted lane markings.

Since going in a circle around the track will lead to predominance of turns in
the direction of driving, the track was driven in both directions

#### Track 1 - Driving Left

![][initial]

#### Track 1 - Driving Right

![][track-1-right]

#### Track 2 - Driving in Both lanes

***Track-2*** is a two-lane mountain track with continuos markings but with a narrow
lane-width. The horizon on this track is in some cases visible, but mostly not. Also
the predominant background are dark-green steep hills, steep rocks and abysses.
The terrain of track 2 is abbundant of steep climbs, drops and sharp turns.

![][track-2]


### Data Preprocessing

The global lighting condition and scenery color distribution varies greatly
between both tracks. To increase the ability of the network to generalize on both
tracks, several pre-processing data augmentations where attempted.

#### Normalizing the steering angle distributions

As can be seen, the steering angle distributions on track-1 depend on the
direction being driven and are mostly concentrated around zero - i.e. around straight-line driving . Since it is necessary to train the car to make turns, as good as driving straight in a line, curved sections must be augmented and repeated.


##### **Using the Side Cameras**
The simulator generates data from three camera locations, centre of car and left/right
translated cameras. The left-right translation was assumed to impact the steering angle
by a parametric threshold (0.25 < threshold < 0.35). This effectively tripples the
dataset by solely adding images that simulate curves.

![][track-1-left]

##### **Image Flipping**
Sections of the data with a greater steering angle then a threshold (0.1 < threshold < 0.4) where flipped and their corresponding steering angles inverted.

**Example**: Track-1 Left driving flipped:
![][track-1-left-flipped]

**Example**: Track-1 Right driving flipped:
![][track-1-right-flipped]

As can be seen, image distributions now are approximately symmetrical mirror images
of each other round the 0 axis.

##### **Combining different runs**

In order to improve upon the left-right driving differences, runs can be combined.

![][combined]

##### **Reducing Zero Angle Images**

Since zero-angle images are still the most abbundant, their probability of occurrence
was reduced to a parametric threshold (0.2 < threshold < 0.4)

![][combined-reduced-centre]

#### Adding random noise and modifiying image brightness

Due to the large difference in backgrounds, methods for global lighting condition
and background generalizations were seen as potentially benefitial.

##### Random noise addition

Images where augmented with random noise, through which each image pixel gets added or
substracted a random value picked from a normal distribution with a parametric absolute
maximum predefined.

![][with-and-without-noise]

##### Brightness modification

The global pixel values of images got added or subtracted a random value taken from
a normal distribution with parametric absolute maximum and truncated to a uint8
compatible size.

![][with-modified-brightness]

## Model

![][model]

## Results

## Conclusion

## References
[1] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner,B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller,J. Zhang, et al. ***End to end learning for self-driving cars***, arXiv preprint arXiv:1604.07316, 2016
