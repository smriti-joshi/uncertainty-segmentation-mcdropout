import os
from nnunetv2.evaluation.evaluate_predictions import compute_metrics #nnunet-pipeline
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO #nnunet-pipeline
import csv
import numpy as np
from skimage.measure import find_contours
from scipy.spatial import cKDTree
import SimpleITK as sitk


def hausdorff_distance_mask(image0, image1, method = 'standard'):

    # https://github.com/scikit-image/scikit-image/issues/6890
    """Calculate the Hausdorff distance between the contours of two segmentation masks.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a pixel from a segmented object. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of the segmentation mask contours in
        ``image0`` and ``image1``, using the Euclidean distance.
    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on the 
    contour of ``image0`` and its nearest point on the contour of ``image1``, and 
    vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    Examples
    --------
    >>> ground_truth = np.zeros((100, 100), dtype=bool)
    >>> predicted = ground_truth.copy()
    >>> ground_truth[30:71, 30:71] = disk(20)
    >>> predicted[25:65, 40:70] = True
    >>> hausdorff_distance_mask(ground_truth, predicted)
    11.40175425099138
    """
    image0_array = sitk.GetArrayFromImage(sitk.LabelContour(sitk.ReadImage(image0, sitk.sitkUInt8)))
    image1_array = sitk.GetArrayFromImage(sitk.LabelContour(sitk.ReadImage(image1, sitk.sitkUInt8)))

    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    
    a_points = np.argwhere(image0_array>0)
    b_points = np.argwhere(image1_array>0)
    
    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))
    
def compute_average_dice_and_iou(path_to_pred, text='dummy'):
    path_to_gt = '/workspace/ground-truth'
    
    dice = 0
    iou = 0
    haus = 0
    counter = 0
    # with open('metrics_pre.csv') as file:
    for pred_id in os.listdir(path_to_pred):
        if pred_id.endswith('.nii.gz'):
            gt_path = os.path.join(path_to_gt, pred_id)
            pred_path = os.path.join(path_to_pred, pred_id)
            results = compute_metrics(gt_path, pred_path, image_reader_writer=SimpleITKIO(), labels_or_regions= [0, 1])
            haus_distance =  hausdorff_distance_mask(pred_path, gt_path, method='modified')
            dice = dice + results['metrics'][1]['Dice']
            iou = iou + results['metrics'][1]['IoU']
            haus = haus + haus_distance
            counter = counter + 1
    av_dice = dice/counter
    iou = iou/counter
    haus = haus/counter

    print('--------------------------------------------------------------------------------')
    print(text, '\nAverage dice:' , av_dice, '\nAverage iou:', iou, '\nAverage Hausdorff:', haus)
    return av_dice, iou, haus

def compute_average_dice_and_iou_with_patient_id(path_to_pred, patient_ids, text = 'Dummy'):
    path_to_gt = '/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw/Dataset203_Pseudolabels/Tests/Preprocessed_single_breasts/labels'
    
    dice = []
    haus = []
    
    for pred_id in patient_ids:
        if pred_id.endswith('.nii.gz'):
            gt_path = os.path.join(path_to_gt, pred_id)
            pred_path = os.path.join(path_to_pred, pred_id)
            result = compute_metrics(gt_path, pred_path, image_reader_writer=SimpleITKIO(), labels_or_regions= [0, 1])
            haus_distance =  hausdorff_distance_mask(pred_path, gt_path, method='modified')
            haus.append(haus_distance)
            dice.append(result['metrics'][1]['Dice'])
            
    haus = np.array(haus)
    dice = np.array(dice)

    print('--------------------------------------------------------------------------------')
    print(text, '\nAverage dice:' , np.mean(dice), '\nAverage Hausdorff:', np.mean(haus))
    
    return dice, haus