from statistical_significance import StatTest
from compute_metrics import compute_average_dice_and_iou_with_patient_id # from nnunet-code
import os
import numpy as np

class All_Metrics():
    def __init__(self, ids) -> None:
        self.stats = StatTest(ids)
        self.ids = ids
    
    def compute_one(self, pred_path, text):
        dice, haus = compute_average_dice_and_iou_with_patient_id(pred_path, self.ids, text)
       
        self.stats.calculate_wilcoxon_quick(dice, 'dice')
        self.stats.calculate_wilcoxon_quick(haus, 'hausdorff')

    def compute_all(self, root_path):
        models = ['five_models', 'ten_models', 'fifteen_models']
        categories = ['hard_voted', 'soft_voted', 'high_certainty', 'higher_certainty', 'highest_certainty']

        for model in models:
            print('-----------------------------------------------------------------------------------------')
            print(model+'\n')
            for category in categories:
                self.compute_one(os.path.join(root_path, model, category), category)

def main():
    
    # all cases
    test_path_02 = '/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d_dropouts/3d_dropouts/with_probabilities_02'
    patient_ids = np.array(sorted(os.listdir(os.path.join(test_path_02, 'five_models', 'hard_voted'))))
    metrics_calculator = All_Metrics(patient_ids)
    metrics_calculator.compute_all(test_path_02)

    # easy cases
    test_path_02 = '/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d_dropouts/3d_dropouts/with_probabilities_02'
    patient_ids = np.array(sorted(os.listdir(os.path.join(test_path_02, 'five_models', 'hard_voted'))))
    easy_ids = [2, 5, 6, 9, 11, 12, 13, 14, 15, 16, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29] # greater than equals to 3
    patient_ids = patient_ids[easy_ids]
    metrics_calculator = All_Metrics(patient_ids)
    metrics_calculator.compute_all(test_path_02)

    # # hard - fuzzy cases
    test_path_02 = '/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d_dropouts/3d_dropouts/with_probabilities_02'
    patient_ids = np.array(sorted(os.listdir(os.path.join(test_path_02, 'five_models', 'hard_voted'))))
    hard_ids = [0, 1, 3, 4, 7, 8, 10, 17, 18, 20]
    patient_ids = patient_ids[hard_ids]
    metrics_calculator = All_Metrics(patient_ids)
    metrics_calculator.compute_all(test_path_02)