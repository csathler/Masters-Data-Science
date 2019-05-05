Main Programs 
=============

+----------------+-----------------------------------------------------------------------------+
| Program        | Purpose                                                                     | 
+================+=============================================================================+
|                | - Load data for the project                                                 | 
| load_data.py   | - Output:                                                                   | 
|                | - discussion tree information, discussion nodes, comments; all csv format   |
+----------------+-----------------------------------------------------------------------------+ 
|                |                                                                             |
| question0.py   | - Output: displays training and test dataset statistics                     | 
|                |                                                                             |
+----------------+-----------------------------------------------------------------------------+ 
|                | - Answer "Was the data loaded properly?"                                    |
| question1.py   | - Output:                                                                   | 
|                | - question1.png                                                             |
+----------------+-----------------------------------------------------------------------------+ 
|                | - Answer "Are certain users more persuasive than others?"                   | 
| question2.py   | - Output:                                                                   | 
|                | - question2_fig1.png and question2_fig2.png                                 |
+----------------+-----------------------------------------------------------------------------+ 
|                | - Answer "Are certain users more susceptible to persuasion?"                |
| question3.py   | - Output:                                                                   | 
|                | - question3_fig1.png and question3_fig2.png                                 |
+----------------+-----------------------------------------------------------------------------+ 
|                | - Answer "Is persuasion more likely at certain times of day or week?"       |
| question4.py   | - Output:                                                                   | 
|                | - question4.png                                                             |
+----------------+-----------------------------------------------------------------------------+ 
|                | - Answer "Is there relationship between tree shape and persuasion?"         |
| question5.py   | - Output:                                                                   | 
|                | - question5.png                                                             |
+----------------+-----------------------------------------------------------------------------+ 
|                | - Run predictive analytics                                                  |
| predict.py     | - Output:                                                                   |
|                | - predictive analytics results including metrics                            |
+----------------+-----------------------------------------------------------------------------+ 


Supporting Functionality
========================

+------------------------+-----------------------------------------------------------------------------+
| Program                | Purpose                                                                     | 
+========================+=============================================================================+
|                        |                                                                             |
| util_load_data_call.py | - Implement call interface for load_data.py (see load_data.py Usage below)  |
|                        |                                                                             |
+------------------------+-----------------------------------------------------------------------------+
|                        |                                                                             |
| util_download_data.py  | - Implement functionality to fetch project raw data from Cornell website    |
|                        |                                                                             |
+------------------------+-----------------------------------------------------------------------------+
|                        |                                                                             |
| util_download_data.py  | - Allow visualization of discussion tree in indented format (see below)     |
|                        |                                                                             |
+------------------------+-----------------------------------------------------------------------------+


Log files
=========

Two log files are posted to the log directory.

load_data.log was created from run of full data load. See link to file below:

https://gitlab.com/cloudmesh_fall2016/project-016/blob/master/code/logs/load_data.log

predict.log was created from run of predictive analytics. See link to file below:

https://gitlab.com/cloudmesh_fall2016/project-016/blob/master/code/logs/predict.log


load_data.py Usage
==================

Usage: load_data.py all|train|test pos_int_number_of_discussions_trees_to_load

Examples 1: Load all data

	python load_data.py all 50000

Examples 2: Load all train data

	python load_data.py train 50000

Examples 3: Load all test data

	python load_data.py test 50000

Examples 4: Used for testing - Load 100 discussion trees; train data only

	python load_data.py train 100

Examples 5: Used for testing - Load 100 discussion trees; train and test data

	python load_data.py all 100


Visualization of Discussion Trees
=================================

Example: 
$ python util_disc_trees.py

Output:

Tree with delta ---------------------------

2rosbp level: 0 degree: 3 height: 6 (ChagSC)

    cnhxfvv level: 1 degree: 1 height: 3 (McKoijion)  >>> Delta!

        cnhy6ks level: 2 degree: 2 height: 2 (ChagSC)

            cni6qkd level: 3 degree: 1 height: 1 (piwikiwi)

                cnid4t0 level: 4 degree: 0 height: 0 (TeslaIsAdorable)

            cnhybr1 level: 3 degree: 0 height: 0 (DeltaBot)

    cnhv12w level: 1 degree: 1 height: 5 (gaviidae)

        cnhvo84 level: 2 degree: 1 height: 4 (ChagSC)

            cnhw0tz level: 3 degree: 2 height: 3 (gaviidae)

                cnhwfls level: 4 degree: 3 height: 2 (Ibnalbalad)

                    cnhwrhw level: 5 degree: 1 height: 1 (nutelly)

                        cnhx0px level: 6 degree: 0 height: 0 (Ibnalbalad)

                    cnhy1t4 level: 5 degree: 0 height: 0 (gaviidae)

                    cni1fgg level: 5 degree: 0 height: 0 (spazdor)

                cnhxa25 level: 4 degree: 1 height: 1 (princessbynature)

                    cnhy4jj level: 5 degree: 0 height: 0 (gaviidae)

    cninihg level: 1 degree: 0 height: 0 (vey323)

