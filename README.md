# TutVis
![](UI/images/interface.PNG?raw=true)

Online text and video tutorials are among the most popular and heavily used resources for learning feature-rich software applications (e.g., Photoshop, Weka, AutoCAD, Fusion 360). However, when searching, users can find it difficult to assess whether a tutorial is designed for their level of expertise. TutVis stands for tutorial visualization, which is a Photoshop tutorial browsing system that provides auto-generated information to assist users in the tutorial searching and selection. The provided information is as follows: difficulty level (advanced/beginner), topics covered, length, text complexity, command usage ratio. This information is automatically calculated using different supervised and unsupervised machine learning algorithms.

• The browsing interface is developed using angular js, node js, d3 js, chart js, and bootstrap.<br/>
• The training data (1300 text and video tutorials of Photoshop and Fusion 360) are crawled from 13 different sources. <br>
• The information/features are generated (feature engineering) using the topic modeling algorithm (LDA), readability formulas, and heuristic calculations. <br/>
• For the classification of the tutorial’s difficulty Random Forest algorithm is used.<br/>
• The classification model can classify a tutorial's difficulty (advanced/beginner) with 85% accuracy.

# UI directory
This directory contains the project for the web interface of the prototype which visualizes the auto generated features from the machine learning model.

# Plot directory
This directory contatins scripts to visualize different aspects and exploratory data analysis of the model.

# Learning Curve directory
This directory contatins a script to build the learning curve for this model.

# Feature generation directory
This directory incldues all the scripts to generate tutorial's features (i.e., feature engineering) that have been used to train the model.

# Classification directory
This directory includes the scripts to train_test using random forest, grid search for hyperparameter optimazation and feature importance visualization.

# Calculation of topic coherence and perplexity directory
This directory includes scripts to calculate the coherence and perplexity of the topic model.
