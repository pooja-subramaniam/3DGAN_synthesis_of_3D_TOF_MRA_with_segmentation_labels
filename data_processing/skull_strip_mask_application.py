import nibabel as nib
import os


# Source folder for extracted masks
root_folder = '/data-raid5/pooja/3DGAN'
dataset = '1kplus'  # 'PEGASUS_denoised' #
probs_dataset = '1kplus'  # 'PEGASUS'  #
split = 'test'
label_type = 'prediction'  # 'ground_truth' #
ground_truth_label_name = '001_Vessel-Manual-Gold-int.nii'
if dataset == '1kplus':
    mask_fn = '002_NLM-Ric-denoised_NUC_BET-fsl.mask.nii'
else:
    mask_fn = '001_NUC_BET-fsl.mask.nii'

# Folder containing ground_truth/prediction labels for mask application
if label_type == 'ground_truth':
    root_folder_segmentation = os.path.join(root_folder, 'Data', dataset,
                                            split)
else:
    trial_name = 'real_1e-04'
    threshold = 0.5
    root_folder_segmentation = os.path.join(root_folder, 'Segmentation',
                                            'Results', trial_name, 'unet-3d',
                                            'probs',
                                            f'test_{probs_dataset}_{threshold}'
                                            )
    # new folder for the skull-stripped images/labels to be saved if necessary
    if not os.path.exists(root_folder_segmentation+'_ss'):
        os.mkdir(root_folder_segmentation + '_ss')

# all patient ids within the dataset and split from source folder
patients = os.listdir(os.path.join(root_folder, 'Data', dataset, split))

## go through all patients and apply mask on ground truth and the prediction
for pat in patients:
    print(f"Loading patient data of {pat}")
    if label_type == 'ground_truth':
        label_nif = nib.load(os.path.join(root_folder, 'Data', dataset, split,
                                          pat, ground_truth_label_name)
                             )
        label = label_nif.get_fdata()

    else:
        label_nif = nib.load(os.path.join(root_folder_segmentation,
                                          'probs_' + pat + '_.nii')
                             )
        label = label_nif.get_fdata()
    try:
        mask = nib.load(os.path.join(root_folder, 'Data', dataset, split, pat,
                                     mask_fn + '.gz')
                        ).get_fdata()
    except:
        mask = nib.load(os.path.join(root_folder, 'Data', dataset, split, pat,
                                     mask_fn)
                        ).get_fdata()

    label[mask == 0] = 0
    new_label_nifti = nib.Nifti1Image(label, label_nif.affine)
    print(f'Saving new skull-stripped label of patient {pat}')
    if label_type == 'ground_truth':
        print(ground_truth_label_name.split('.')[0])
        nib.save(new_label_nifti, os.path.join(root_folder, 'Data', dataset,
                                               split, pat,
                                               ground_truth_label_name.split(
                                                   '.')[0] + '_ss.nii')
                 )
    else:
        nib.save(new_label_nifti, os.path.join(root_folder_segmentation
                                               + '_ss', 'probs_' + pat
                                               + '_.nii'))

print('Saved all new skull-stripped data')
