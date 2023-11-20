## [Leveraging Epistemic Uncertainty to Improve Tumour Segmentation in Breast MRI: An exploratory analysis]()

In SPIE Medical Imaging 2024.

![Contribution](images/combination-dropout.jpg)

## Dataset
A subset of the [Duke Dataset](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) was for this analysis. The whole dataset is publicly available at [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903).

The train-validation-test splits can be found [here] (segmentation/modifications_nnunet/utils/train-val-test-split.json).

Bias field correction was performed on the original images ([here] (segmentation/preprocessing/preprocess_niftis.py)), followed by selection of laterality which contains the tumor ([here] (/preprocessing/convert_to_unilateral.py)). 

The segmentation was performed using [nnUnet framework](https://github.com/MIC-DKFZ/nnUNet). The modification to incorporate dropout in the pipeline is [here](segmentation/modifications_nnunet/get_network_from_plans.py). The segmentations used in the study were kindly provided by the authors of [Caballo et al](https://doi.org/10.1002/jmri.28273).
 
### Combination by voting 
![Voting Scheme](images/combination-dropout.jpg)

The code of combining segmentation with difference voting schemes can be found [here] (combination/voting.py)

### Evaluation

The script for segmentation metrics (Dice coefficient, Hausdorff distance) and statistical significance (Wilcoxon signed rank test) are [here](evaluation/compute_metrics_with_stats.py). 
