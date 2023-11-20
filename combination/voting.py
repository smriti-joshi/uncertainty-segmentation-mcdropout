import os
import SimpleITK as sitk
import numpy as np

def save_image(save_array, reference_img, save_path):
    image = sitk.GetImageFromArray(save_array)
    image.SetDirection(reference_img.GetDirection())
    image.SetSpacing(reference_img.GetSpacing())
    image.SetOrigin(reference_img.GetOrigin())
    
    sitk.WriteImage(image, os.path.join(save_path))

class Voting():

    def __init__(self) -> None:
        self.output_path = '/workspace/output'
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.model_path = '/workspace/inputs_models'
        self.models = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5'] # extend to 10 models, 15 models
    
    def hard_voting(self):
        output_path = os.path.join(self.output_path, 'hard_voted')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
       
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        for patient_id in patient_ids:
            if patient_id.endswith(".nii.gz"):
                seg = None
                for model in self.models:
                    if seg is None:
                        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.model_path, model, patient_id)))
                    else:
                        seg += sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.model_path, model, patient_id)))

                hard_voted = np.float64(seg > (int(len(self.models)/2) + 1))
                save_image(hard_voted, sitk.ReadImage(os.path.join(self.model_path, model, patient_id)), os.path.join(output_path, patient_id))

    def soft_voting(self):
        output_path = os.path.join(self.output_path, 'soft_voted')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        for patient_id in patient_ids:
            if patient_id.endswith(".npz"):
                seg = None
                for model in self.models:
                    if seg is None:
                        seg = np.load(os.path.join(self.model_path, model, patient_id))['probabilities']
                    else:
                        seg += np.load(os.path.join(self.model_path, model, patient_id))['probabilities']
                
                avg = seg/len(self.models)
                soft_voted = np.argmax(avg, axis=0)
                save_image(soft_voted, sitk.ReadImage(os.path.join(self.model_path, model, patient_id[:-4]+'.nii.gz')), os.path.join(output_path, patient_id[:-4]+'.nii.gz'))

    def high_certainty(self, certainty=0.1, text = 'high_certainty'):
        output_path = os.path.join(self.output_path, text)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        for patient_id in patient_ids:
            if patient_id.endswith(".npz"):
                seg = None
                for index, model in enumerate(self.models):
                    if seg is None:
                        shape = np.load(os.path.join(self.model_path, model, patient_id))['probabilities'].shape
                        seg = np.empty((len(self.models), shape[0], shape[1], shape[2], shape[3]))
                    seg[index] = np.load(os.path.join(self.model_path, model, patient_id))['probabilities']
                    # else:
                    #     seg += np.load(os.path.join(model_path, model, patient_id))['probabilities']
                
                mean_seg = np.mean(seg, 0)
                mean_std_dev = np.std(seg, 0)

                segmentation = np.argmax(mean_seg, axis=0).astype(bool)
                std = np.zeros_like(segmentation).astype(np.float32)
                std[segmentation] = mean_std_dev[1][segmentation]
                std[~segmentation] = mean_std_dev[0][~segmentation]
                
                high_certainty = np.argmax(mean_seg, axis=0) * np.int8(std<certainty)

                save_image(high_certainty, sitk.ReadImage(os.path.join(self.model_path, model, patient_id[:-4]+'.nii.gz')), os.path.join(output_path, patient_id[:-4]+'.nii.gz'))

def main():
    
    voter = Voting()
    voter.soft_voting()
    print('soft_voting done!')
    voter.hard_voting()
    print('hard_voting done!')
    voter.high_certainty(certainty=0.1, text='high_certainty')
    print('high uncertainty done!')
    voter.high_certainty(certainty=0.05, text='higher_certainty')
    print('higher uncertainty done!')
    voter.high_certainty(certainty=0.02, text='highest_certainty')
    print('highest uncertainty done!')

