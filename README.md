# multimodal-DDU
Multimodal density estimation for deep deterministic uncertainty.

# Code description.
- Files for running the experiments
  - train_models.py: Script for training models. 
  - single_run.sh: Convenience script for training models.
  - multi_run.sh: Convenience script for training models (this one is useful for multiple runs on different datasets).
  - run_tests.py: Script for running non-ensemble tests.
  - run_tests_ensemble.py: Script for running ensemble tests.
  - run_tests_ensemble.sh: Convenience script for running all ensemble tests.
  - vis_toy_dataset.py: Script for visualizing the uncertainty on a toy (swirl) dataset.
- Model code
  -  WRN.py: Wide-ResNet implementation.
  -  ensembles.py: Ensemble implementation.
  -  resNet.py: ResNet implementation.
  -  resNetSNGP.py: SNGP ResNet implementation.
  -  vis_models.py: Models used in toy dataset visualization (vis_toy_dataset.py).
- Uncertainty
  - uncertainty.py: Implementation of uncertainty estimation methods.
- Testing scripts (for debugging)
  - test_ensembles.py
  - test_resnet.py
  - test_sngp.py
  - test_wrn.py 
