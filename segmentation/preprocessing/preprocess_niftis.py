import os
import SimpleITK as sitk
import concurrent.futures
import time

def preprocess_breast_mri(nifti_mri, output_filepath=None, resample=True, spacing=[1, 1, 1], normalize=False, bias_correction=True, shrink_factor=4):   
    
    image_itk = sitk.ReadImage(nifti_mri, sitk.sitkFloat32)

    if bias_correction:
    # N4BiasFieldCorrectionImageFilter takes too long to run, shrink image
        mask_breast = sitk.OtsuThreshold(image_itk, 0, 1)
        shrinked_image_itk = sitk.Shrink(image_itk, [shrink_factor] * image_itk.GetDimension())
        shrinked_mask_breast = sitk.Shrink(mask_breast, [shrink_factor] * mask_breast.GetDimension())
        # sitk.WriteImage(mask_breast, os.path.join(output_folder_mris, patient_id + '_mask.nii.gz'))
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(shrinked_image_itk, shrinked_mask_breast)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_itk)
        # sitk.WriteImage(log_bias_field, os.path.join(output_folder_mris, patient_id + '_log_bias_field.nii.gz'))
        corrected_image_itk = image_itk / sitk.Exp(log_bias_field)
        image_itk = corrected_image_itk

    if normalize:
        sitk.RescaleIntensity(image_itk, 0, 255)

    if output_filepath:
        sitk.WriteImage(image_itk, output_filepath)

    return image_itk


def create_processed_data(patient_id):
    input_data_folder = '/data/Duke-Breast-Cancer-MRI-Nifti-Whole'
    output_data_folder = '/data/Duke-Breast-Cancer-MRI-Nifti-Whole-Preprocessed'
 
    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)

    if not os.path.exists(os.path.join(output_data_folder, patient_id)):
        os.mkdir(os.path.join(output_data_folder, patient_id))

    for phase in os.listdir(os.path.join(input_data_folder, patient_id)):
        preprocess_breast_mri(nifti_mri=os.path.join(input_data_folder, patient_id, phase),
                              output_filepath=os.path.join(output_data_folder, patient_id, phase))
    