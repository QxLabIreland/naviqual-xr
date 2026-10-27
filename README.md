# NAVIQUAL-XR: Finding the Sweet Spot of a Live Music Environment in Augmented Reality
This repository contains the code to reproduce the results from the paper NAVIQUAL-XR: Finding the Sweet Spot of a Live Music Environment in Augmented Reality. This work is currently under review in ACM Transactions on Multimedia Computing, Communications, and Applications (ACM TOMM).

## What is NAVIQUAL-XR?
NAVIQUAL-XR is a framework to identify the spot with the best quality of experience in a given live music environment. The system overview of NAVIQUAL-XR is shown in the following figure. Participants wearing augmented reality (AR) glasses were asked to find the sweet spot in different live music scene configurations. These sweet spot selections served as the ground truth for evaluating the objective metrics that predicted these sweet spots.

<p align="center">
  <img src="output/naviqual-xr-diagram.jpg" width="100%" />
  <br>
  <em>Figure 1: System overview of evaluating NAVIQUAL-XR</em>
</p>

## Setup

## Navigation Behavior
* Run `python 1_analyze_navigation.py` to analyze the participant trajectories under different scene conditions.
* Some figures are shown here to compare trajectories between between audio-only and audio-visual conditions.
* The full set of figures for six auditory and six audio-visual scenes are located in [here](./output/1_navigation_behaviour/).

<p align="center">
  <img src="output/1_navigation_behaviour/S1-A_grid_distribution.jpg" width="31%" />
  <img src="output/1_navigation_behaviour/S3-A_grid_distribution.jpg" width="31%" />
  <img src="output/1_navigation_behaviour/S5-A_grid_distribution.jpg" width="31%" />
  <br>
  <img src="output/1_navigation_behaviour/S1-AV_grid_distribution.jpg" width="31%" />
  <img src="output/1_navigation_behaviour/S3-AV_grid_distribution.jpg" width="31%" />
  <img src="output/1_navigation_behaviour/S5-AV_grid_distribution.jpg" width="31%" />
  <br>
  <em>Figure 2: Participant navigation trajectories from the initial position to selected subjective sweet spots. The top and bottom rows depict audio-only and audio-visual environments, respectively. Audio-visual conditions yielded direct navigation paths, whereas auditory conditions prompted broader spatial exploration.</em>
</p>

## Participant Sweet Spot Selections
* Run `python 2_analyze_sweetspots.py` to generate the participant sweet spot selections for all twelve scenes.
* Some figures are shown here to compare sweet spot selections under different scene conditions.
* The full set of figures are located in [here](./output/2_participant_sweetspots/).

<p align="center">
  <img src="output/2_participant_sweetspots/S2-AV_sweetspots.jpg" width="31%" />
  <img src="output/2_participant_sweetspots/S4-AV_sweetspots.jpg" width="31%" />
  <img src="output/2_participant_sweetspots/S6-AV_sweetspots.jpg" width="31%" />
  <br>
  <em>Figure 3: Subjective sweet spot locations and directions across three audio-visual scenes. Consensus sweet spots shift away from speech distractors while maintaining preferred acoustic balance of the music sources.</em>
</p>

## Sweet Spots under Auditory vs Audio-Visual Modalities
* Run `python 3_analyze_modalities.py` to generate intra-participant sweet spot selections under two cue modalities.
* The full set of figures for all six scene conditions are located in [here](./output/3_cue_modalities/).

<p align="center">
  <img src="output/3_cue_modalities/S1_modalities.jpg" width="31%" />
  <img src="output/3_cue_modalities/S3_modalities.jpg" width="31%" />
  <img src="output/3_cue_modalities/S5_modalities.jpg" width="31%" />
  <br>
  <em>Figure 4: Participant sweet spot selections in auditory vs audio-visual modalities with increasing number of distractor sources. Sweet spot shifts become more pronounced as distractor sources increase, demonstrating that visual cues intensified participant avoidance from distractor sources.</em>
</p>

## Subjective Sweet Spot Maps
* Run `python 4_subjective_maps.py` to generate the subjective sweet spot maps with respect to listener location and direction. 
* These animations show how the subjective sweet spots shift with respect to the listener direction.
* The full set of animations for six auditory scenes and six audio-visual scenes are located in [here](./output/4_subjective_maps/).

<p align="center">
  <img src="output/4_subjective_maps/S1-AV_subjective_map.gif" width="70%" alt="Model Animation Demo">
  <br>
  <em>Figure 5: Subjective sweet spot maps with respect to listener location and direction. Participant sweet spot locations shift with respect to listener direction. Generally, participant sweet spots shift toward the right as participant heading rotates counterclockwise, such that participants maintain their direction toward the music sources.</em>
</p>


## Objective Sweet Spot Maps

## Subjective vs Objective Sweet Spot Maps

## Similarity Metrics
