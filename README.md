# Diffusion-based Time Series Imputation with DPS Guidance
This project explores diffusion-based imputation for multivariate time series data.
Starting from the CSDI baseline, I investigated a DPS-style guidance method to better recover high-frequency patterns in missing region


## Project Summary
- **Task**: Multivariate time series imputation
- **Baseline**: CSDI
- **My contribution**: DPS-style guided sampling
- **Focus**: Recovering difficult or high-frequency missing patterns
- **Tools**: Python, PyTorch


## Problem
Missing values frequently occur in real-world multivariate time series data (e.g., healthcare, air quality).  
Although diffusion-based models such as CSDI perform strong imputation overall, they can struggle to recover local or high-frequency patterns in missing regions.


## Method Overview
This project builds on CSDI as a baseline diffusion model and introduces a DPS-style guidance mechanism during the sampling stage.

To improve reconstruction in missing regions, the guidance is not applied globally but restricted to a local temporal window around each missing segment.  
Specifically, `left_window` and `right_window` define the range of observed values adjacent to missing regions, and a masking strategy is used to apply DPS only within this selected region.

In addition, instead of using a single threshold-based frequency selection, the high-frequency components are controlled using a frequency band (defined by `f_low` and `f_high`).  
This design is inspired by FGTI and allows more flexible and stable control of frequency information.

Overall, the method focuses on guiding the model using locally relevant information and controlled frequency bands to better recover complex patterns in missing regions.

## Key Arguments

- `left_window`, `right_window`  
  Define the temporal window around missing regions where DPS guidance is applied.  
  Guidance is restricted to observed values within this window to preserve local structure.

- `f_low`, `f_high`  
  Define the frequency band used for guidance.  
  Instead of thresholding, a frequency range is used to better control high-frequency components.

- `dps_scale`  
  Controls the strength of guidance during sampling.

## Requirement
Please install the packages in requirements.txt

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```
### Download the air quality dataset 
```shell
python download.py pm25
```


## Experiments 

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio_dps.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples] --left_window[window_size] --right_window[window_size] --f_low[frequency low cut] --f_high[frqunecy high cut] --dps_scale[guidance strength]
```

### training and imputation for the pm25
```shell
python exe_pm25_dps.py --nsample [number of samples] --validationindex[0-4] --left_window[window_size] --right_window[window_size] --f_low[frequency low cut] --f_high[frequency high cut] --dps_scale[guidance strength]
```


### Visualize results
After running experiments, results can be visualized using the provided scripts.
'visualize_examples.ipynb' is a notebook for visualizing results.


