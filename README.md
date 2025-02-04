## Contrast Synthesis with Uncertainty Estimation using Quantile Regression and U-Nets

This research presents a novel approach to synthesize missing MRI contrasts with uncertainty estimation using a U-Net framework based on quantile regression. Multi-contrast brain MRI data is vital for comprehensive analyses in neurological conditions such as traumatic brain injury (TBI) and epilepsy, as different MRI contrasts provide complementary information on brain structure and pathology. However, challenges in clinical settings often lead to incomplete multi-contrast MRI datasets, which complicates downstream analyzes, particularly in deep learning applications that require consistent data across all contrasts. Our method addresses this issue by generating missing MRI contrasts from available contrasts while quantifying the uncertainty of these synthesized images. We employ a U-Net architecture with quantile regression to predict both the missing contrasts and the associated uncertainty, providing more reliable synthetic images for clinical and research applications.

### How to run
The relevant code is inside `src/` folder.
1. Make sure you have all dependencies installed. You may run `conda env create -f reqs.yml` to set them up.
2. You may convert notebooks to python code if needed.

### Code
1. Run `main` to train Unet-synth model
2. Run `save_results` to save the synthetic data to disk
3. Run `tumseg` to load this synthetic data for brain tumor segmentation task

### Analysis
1. Run `analyse_results` for analysing training (train-loss plots) and to inspect your results.

