from scipy.stats import wilcoxon
from compute_metrics import compute_average_dice_and_iou_with_patient_id

class StatTest():
    def __init__(self, patient_ids=None):
        self.baseline = '/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset204_DukePhaseOneHalf/Tests/3d_dropouts/3d_dropout_best_checkpoint_1000_02'
        self.test_ids = patient_ids
        
        self.array_baseline_dice, self.array_baseline_haus = compute_average_dice_and_iou_with_patient_id(self.baseline, self.test_ids, 'Baseline Metrics:')
        print('\n---------------------------------------------------------------------------')

    def calculate_wilcoxon_quick(self, array_prediction, metric='dice'):
        if metric == 'dice':
            res = wilcoxon(array_prediction, self.array_baseline_dice)
        else:
            res = wilcoxon(array_prediction, self.array_baseline_haus)
        
        print('Wilcoxon for ' + metric + ':', res.pvalue)

