# Strategies for tackling the class imbalance problem of oropharyngeal primary tumor segmentation on Magnetic Resonance Images
The purpose of this work is to investigate different strategies to tackle the class imbalance problem in the oropharyngeal tumor subsite.

Published artcle: https://www.sciencedirect.com/science/article/pii/S2405631622000744

# Walkthrough of the code

The configuration file can be found in config.py. It can be used to change the hyperparameters for training or inference.

The main file can be found in main.py. It is used to run the experiments as defined in the configuration file.

Inside the directory "tools" you can find the scripts needed during the training:

Model_factory: Script that loads the models and performs training, prediction and training
loaders: loaders*.py. Each loader is used differently depending on the config file.
- loader_mp.py is to run the full end-to-end approaches.
- loader_mp_boxes.py to run the segmentation stage from the two-stage approach
- loader_mp_boxes_inf.py to run the segmentation stage in inference time from the boxes define in the config file

The different loss functions can be found in: metrics.py and focal_loss_functions.py
