# tutVis
A photoshop tutorial browsing system where each tutorial is annotated with the tutorial’s difficulty (advanced/ beginner), topics, length, text difficulty, and commands usage. The information presented on the browsing interface is automatically calculated using different machine learning algorithms.<br/>
• The browsing interface is developed using angular js, node js, d3 js, chart js, and bootstrap.<br/>
• The features are generated using the topic modeling algorithm LDA, readability formulas and heuristic calculations. <br/>
• The machine learning part has been developed using python (scikit-learn, gensim). <br/>
• For the classification of the tutorial’s difficulty Random Forest algorithm is used.<br/>

❮img src="tutVis.png" width="100" ❯

# UI folder
contains the web interface of the prototype which visualizes the auto generated features from the machine learning model.

# Plot 
contatins scripts to visualize different aspects of the model.

# LearningCurve folder 
contatins script to build a learning curve for this model.

# Feature Generation 
incldues all the scripts to generate tutorial's features (i.e., feature engineering) that have been used to train the model.

# Classification 
includes the scripts to train_test using random forest, grid search for hyperparameter optimazation and feature importance visualization.

# Calculation of topic coherence and perplexity 
includes scripts to calculate the coherence and perplexity of the topic model.
