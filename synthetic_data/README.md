# Synthetic data
Code for testing and generating synthetic data that can be used for training.

# TODO
  1. Make a script/function to create synth stacks easily

# Files
  * `stats_*.pkl`: Pickle histograms representing the pixel intensity of real data in 256 bins (0->255). '*' should be the date in yymmdd format of the last update. In order, the pickled objects are:
    * `pixel_bkg`: histogram of background pixel intensity
    * `pixel_fg`: histogram of foreground (=ROI) pixel intensity
    * `roi_max`: histogram of maximal pixel intensity among ROI
    * `roi_ave`: histogram of average pixel intensity among ROI
    * `roi_med`: histogram of median pixel intensity among ROI
    * `roi_q75`: histogram of 75th-percentile pixel intensity among ROI
  * `synthetic_generation.ipynb`: notebook used to create the synthetic data
  * `synthetic_tests.ipynb`: notebook used to test the synthetic data generation
