# Deep Learning
This folder contains all the code to test and learn the deep networks for the ROI detection.

# Files
  * `ROI_detection_DL.ipynb`: jupyter notebook used to perform test, and visualize results of the deep networks
  * `run_gridsearch.py`: script to launch gridsearches, using the `run_train.py` script (the code should be modified in order to change parameters and such)
  * `run_train.py`: script to fully train a network. This script can be launched through command line with arguments, or the `main` function can be used in Python code
  * `utils_common`: link to the common utils library
  * `utils_data.py`: useful functions for data manipulation with PyTorch
  * `utils_loss.py`: useful functions for losses/metrics use with PyTorch
  * `utils_model.py`: useful functions for model manipulation with PyTorch (contains the model definition)
  * `utils_test.py`: useful function for testing model with PyTorch
  * `utils_train.py`: useful functions for training model with PyTorch
