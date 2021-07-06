# For reproducibility
seed = 990

# metric options
metrics = ['FID', 'PRD']

# the experiment, epoch and threshold to be used for generated images
gan_trial_num = 16
gan_epoch_num = 99
gan_batch = 1
gan_label_threshold = 0.3

# real data specification
dataset = 'PEGASUS_denoised'
split = 'train'
ppp = 50
image_size = '128x128x64'
patches_dataset = str(ppp) + '_ppp_' + image_size

save_column_names = ['seed', 'dataset', 'split', 'gan_label_threshold',
                     'gan_trial_num', 'gan_epoch_num', 'FID_score']
save_columns = [seed, dataset, split, gan_label_threshold,
                gan_trial_num, gan_epoch_num, ]


data_root = '/data-raid5/pooja/3DGAN/features/'
#data_real = '/data-raid5/pooja/HPC/Data/' + dataset + '/' \
#            + patches_dataset + '/' + split + '/'
#data_gen = '/data-raid5/pooja/3DGAN/generated/Trial_' + str(gan_trial_num) \
#           + '/gen_images_epoch_' + str(gan_epoch_num) \
#           + '_threshold_' + str(gan_label_threshold) + '_' + str(gan_batch) \
#           + '/' + split + '/'
data_real_features = '/data-raid5/pooja/3DGAN/features/real' \
                      + patches_dataset + '/'
data_gen_features = '/data-raid5/pooja/3DGAN/features/generated/Trial_' \
                    + str(gan_trial_num) + '/epoch_' + str(gan_epoch_num) \
                    + '_threshold_' + str(gan_label_threshold) + '/'

save_evaluations_top = '/data-raid5/pooja/3DGAN/evaluations/'
save_evaluations_folder = save_evaluations_top + 'Trial_' \
                          + str(gan_trial_num) + '_prd_10_epoch_' \
                          + str(gan_epoch_num) + '_threshold_' \
                          + str(gan_label_threshold) + '_check/' + split + '/'

# save_evaluations_folder = save_evaluations_top + 'real' + patches_dataset
save_evaluations = save_evaluations_top + 'evaluations.csv'
