##############
## OVERVIEW ##
##############

This is a data release for the Simulated Outcomes Following Carotid Artery Laceration (SOCAL) dataset version 1.

#################
## DATASET USE ##
#################

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. 
https://creativecommons.org/licenses/by-nc/4.0/

If you plan to publish work that used this dataset in any capacity, please cite this data release as well as the following manuscript:

Kugener et al. "Utility of the Simulated Outcomes Following Carotid Artery Laceration Video Data Set for Machine Learning Applications." JAMA Network Open. 2022;5(3):e223177. doi:10.1001/jamanetworkopen.2022.3177

These are neurosurgical training exercise videos performed on cadaveric models. These simulations contain no patient data, no protected health information, and are not derived from patient care settings. All video is owned by the Department of Neurosurgery at the University of Southern California. Please email any correspondence to guillaumekugener93@gmail.com.

#########################
## DATASET DESCRIPTION ##
#########################

Endoscopic endonasal surgery is a common neurosurgical procedure providing access to the cranial base. One of the key steps of the procedure is identifying and preserving each internal carotid artery (ICA), the main arterial blood supply to each cerebral hemisphere. Injury to the ICA can result in stroke, neurologic injury and/or death. Since this injury occurs in fewer than 0.5% of surgical cases, surgeons may not develop proficiency in hemorrhage control during their training programs or clinical practice. A previously validated, high-fidelity simulator for nationwide training courses from 2017-2020 was used to teach surgeons to manage this rare complication. For each cadaver training exercise, a cadaveric head specimen was perfused with an artificial blood substitute after an endonasal surgical approach was prepared, a deliberate injury to the ICA was made, and surgeons attempted to manage the injury twice during a trial time of 5 minutes per simulated session: once before receiving any coaching, and once after expert coaching. Muscle is accepted as the gold standard for achieving sustained hemostasis in ICA injury. Successful placement of muscle over the injury was therefore considered a successful attempt. This cadaveric training exercise has been previously validated as having exceptionally high realism and transferability to the operating room. Details of set-up, participant demographics and performances have been previously published[1]–[3].

Surgeons were recruited through training exercise set-up at national educational conferences for neurosurgery and otolaryngology, as previously published. Non-identifiable surgeon demographic information was collected (e.g. number of years in training, number of endoscopic cases, etc.) and consent was obtained for intraoperative video recording.

Intraoperative video was taken from the Karl Storz Video Neuro-Endoscope used during each of these trials. The dataset we present in this data release consists of these recordings, which have now been appropriately cut, segmented, and annotated in preparation to be disseminated to the research community following a previously described approach[4]. Videos were recorded at a frame rate of 30 frames per second (fps) and at a resolution of 1280x720 or 1920x1080. The duration of the trials varies from 46 seconds to 5 minutes. Each trial video was downsampled from 30 fps to 1 fps using ffmpeg[5].

We hand-annotated surgical tools in each video frame using bounding boxes. Each frame of the downsampled 1 fps dataset was annotated with bounding boxes using the open-sourced image annotation software VoTT[6]. A bounding box was created for each instance of the following tools in frame: suction, grasper, cottonoid, muscle, string, drill, scalpel, and other non-specified surgical tools. For each instance of a tool in frame, a bounding box was drawn around the tool such that the entirety of the tool was encompassed by the bounding box. In conjunction with trial video recordings, “outcomes data” (e.g. EBL, task success) and demographic data (e.g. training status, confidence) was recorded for each participant.

A complete description of the dataset files and columns can be found in the DATASET FILES section below.

[1] G. Zada, J. Bakhsheshian, M. Pham, M. Minneti, E. Christian, J. Winer, A. Robison, B. Wrobel, J. Russin, W. J. Mack, and S. Giannotta, “Development of a Perfusion-Based Cadaveric Simulation Model Integrated into Neurosurgical Training: Feasibility Based on Reconstitution of Vascular and Cerebrospinal Fluid Systems,” Oper. Neurosurg., vol. 14, no. 1, pp. 72–80, Jan. 2018.
[2] D. A. Donoho, C. E. Johnson, K. T. Hur, I. A. Buchanan, V. L. Fredrickson, M. Minneti, G. Zada, and B. B. Wrobel, “Costs and Training Results of an Objectively Validated Cadaveric Perfusion-based Internal Carotid Artery Injury Simulation During Endoscopic Skull Base Surgery,” Int. Forum Allergy Rhinol., vol. 9, no. 7, pp. 787–794, 2019.
[3] M. Pham, A. Kale, Y. Marquez, J. Winer, B. Lee, B. Harris, M. Minnetti, J. Carey, S. Giannotta, and G. Zada, “A Perfusion-based Human Cadaveric Model for Management of Carotid Artery Injury during Endoscopic Endonasal Skull Base Surgery,” J. Neurol. Surg. Part B Skull Base, vol. 75, no. 5, pp. 309–313, Oct. 2014.
[4] D. J. Pangal, G. Kugener, S. Shahrestani, F. Attenello, G. Zada, and D. A. Donoho, "Technical Note: A Guide to Annotation of Neurosurgical Intraoperative Video for Machine Learning Analysis and Computer Vision," World Neurosurg., Published online March 12, 2021, doi:10.1016/j.wneu.2021.03.022.
[5] “FFmpeg.” [Online]. Available: https://ffmpeg.org/. [Accessed: 03-Dec-2020].
[6] microsoft/VoTT. Microsoft, 2020.

########################
## DATASET STATISTICS ##
########################

Number of surgeons: 177
Number of trials with outcomes data: 365

Number video annotated trials: 147
Number of annotated frames: 31443

Number of tools anntoated
- cottonoid: 10005
- drill: 210
- grasper: 15943
- muscle: 4560
- scalpel: 4
- string: 11917
- suction: 22356
- tool: 76

###################
## DATASET FILES ##
###################

##############
# frames.zip #
##############

Zip file that contains all of the video frames, as jpeg files.

#############
# socal.csv #
#############

Description: 
	Each row in this file corresponds to a single bounding box in a single frame. If a frame has multiple annotated tools, then there will be a row for each tool in view. If a frame has no tools in view, then there will be a single row for this frame where the first column contains the frame file name and the remaining columns are empty. 

	For the coordinate values, the top left corner of the image is considered the origin (0,0). The coordinates are integer pixel values. NOT ALL VIDEOS USE THE SAME RESOLUTION - the particular resolutions for each video is specified within the trial_outcomes.csv.

This file does not have a header. From left to right, the columns correspond to:

frame
	Frame file name (equivalent to the frame column in frame_to_trial_mapping.csv)

x1
	The left x coordinate

y1
	The top y coordinate

x2
	The right x coordinate

y2
	The bottom y coordinate

label
	The tool label for this bounding box. Can be one of 8 labels: suction, grasper, cottonoid, muscle, string, drill, scalpel, tool.

##############################
# frame_to_trial_mapping.csv #
##############################

Description:
	This file provides a mapping from frame file name to the trial ID and frame's position within the video. This file is necessary because some trial recordings were split across multiple videos. This file clarifies how to map all of the images to their appropriate trials.

frame
	Frame file name

trial_id
	Unique identifier for this trial

frame_number
	Index of this frame within the trial video. This is necessary because some trial recordings were split across multiple videos

######################################
# socal_participant_demographics.csv #
######################################

Description:
	This file contains the participant level demographics data for participanting surgeons. Not all participants have complete information as not all participants completed the pre and post cadaveric training exercise surveys.

participant_id
	Participant's (surgeon's) ID

cohort
	Cohort that this participant belongs to

specialty
	The specialty of this participant surgeon

training_status
	Participant surgeon's training level ('Attending' or 'Trainee')

total_years_experience
	Total years of surgical experience

years_as_attending
	Total years as an attending surgeon

years_as_trainee
	Total years spent as a resident

endo_last_12_mo
	Number of endoscopic cases in the last 12 months

cadaver_last_12_mo
	Number of cadaver based cases in the last 12 months

aerobic_min_per_week
	Number of minutes performing aerobic exercise per week

aerobic_amount_last_90_days
	Number days with aerobic exercise within the last 90 days

game_min_per_week
	Number of minutes playing video games per week

game_any_min_per_week
	Number of minutes playing any game per week

prior_real
	Experienced a real carotid artery injury in the past

prior_simulated
	Experienced a simulated carotid artery injury in the past

survey_bleeding_importance
	Response to survey question: On a scale of 1-5, how important is surgeon ability to control bleeding

survey_endonasal_importance
	Response to survey question: On a scale of 1-5 how important is a neurosurgeons ability to perform endoscopic endonasal neurosurgery

general_confidence_pre
	Overall confidence in succeeding in this task prior to the start of the cadaveric training exercise

carotid_confidence_pre
	Confidence in repairing an internal carotid artery injury prior to the start of the cadaveric training exercise

survey_anatomy_real
	Response to survey question: Anatomy is realistic

survey_tissue_real
	Response to survey question: Tissue feel is realistic

survey_depth_perception_real
	Response to survey question: Depth perception is realistic

survey_instrument_application_real
	Response to survey question: Instrument application is realistic

survey_image_graphics_real
	Response to survey question: Image projection and graphics are realistic

survey_model_useful_anatomy
	Response to survey question: This model is useful for teaching anatomy

survey_model_useful_planning
	Response to survey question: This model is useful for teaching surgical planning

survey_model_useful_operative_technique
	Response to survey question: This model is useful for teaching operative technique

survey_model_useful_coordination
	Response to survey question: This model is useful for improving hand-eye coordination

survey_model_useful_overall_training
	Response to survey question: This model is useful as an overall training tool

survey_model_useful_ica_injury_repair
	Response to survey question: This model is useful for teaching carotid artery injury repair

survey_skills_transferable
	Response to survey question: Skills learned are transferrable to the operating room

survey_should_be_incorporated
	Response to survey question: This model should be incorporated into a training curriculum

survey_would_recommend
	Response to survey question: I would recommend this model to trainees

general_confidence_post
	Overall confidence in succeeding in this task after to the completion of the cadaveric training exercise

carotid_confidence_post
	Confidence in repairing an internal carotid artery injury after completion of the start of the cadaveric training exercise

training_success
	Whether this participant is considered a training success: failed their first attempt and succeeded their second

simulation_blood_flow
	Blood flow rate (in mL per 30 seconds) during the cadaveric training exercise

hr_baseline_peak
	The highest recorded heart rate (in beats per minute) of the participant prior to the start of the cadaveric training exercise

hr_baseline_average
	The average recorded heart rate (in beats per minute) of the participant prior to the start of the cadaveric training exercise

hr_trial1_peak
	The highest recorded heart rate (in beats per minute) of the participant during trial 1 of the cadaveric training exercise

hr_trial1_average
	The average recorded heart rate (in beats per minute) of the participant during trial 1 of the cadaveric training exercise

hr_intertrial_peak
	The highest recorded heart rate (in beats per minute) of the participant in between trial 1 and trial 2 of the cadaveric training exercise

hr_intertrial_average
	The average recorded heart rate (in beats per minute) of the participant in between trial 1 and trial 2 of the cadaveric training exercise

hr_trial2_peak
	The highest recorded heart rate (in beats per minute) of the participant during trial 2 of the cadaveric training exercise

hr_trial2_average
	The average recorded heart rate (in beats per minute) of the participant during trial 2 of the cadaveric training exercise

hr_posttrial_peak
	The highest recorded heart rate (in beats per minute) of the participant after to the end of the cadaveric training exercise

hr_posttrial_average
	The average recorded heart rate (in beats per minute) of the participant after to the end of the cadaveric training exercise

############################
# socal_trial_outcomes.csv #
############################

Description:
	This file contains the relevant outcomes for each trial within the dataset. Some trials present in this file do not have corresponding video recordings. Each participant underwent two trials, one before and one after expert coaching, denoted by SXXXT1 vs SXXXT2. A trial was considered successful if surgeons successfully stopped bleeding (achieved hemostasis) with a muscle patch within five minutes. Failure was defined as inability to achieve hemostasis with muscle within five minutes. 

trial_id
	Unique identifier for this trial
participant_id
	Participant's (surgeon's) ID

cottonoid
	Time (in seconds) that the cottonoind was first brought into view

cottonoid_success
	Whether cottonoid was successfully  (1) placed or not (0)

muscle
	Time (in seconds) that the muscle was first brought into view

muscle_success
	Whether muscle was successfully (1) placed or not (0)

tth
	Time (in seconds) to hemostasis. Maximum value should be 300

success
	Whether this trial was successfully completed (1) or not (0)

blood_loss
	Measured blood loss over the course of this trial

trial_video_width
	Pixel width of the frames in this trial

trial_video_height
	Pixel height of the frames in this trial
