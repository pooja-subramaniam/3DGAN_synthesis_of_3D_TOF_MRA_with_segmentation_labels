import numpy as np
import os

import evaluation.prd_score as prd
import evaluation.eval_config as ec
import utils as ut
import evaluation.eval_utils as eval_ut


def main():

    print(f'\nEvaluating GAN from Trial {ec.gan_trial_num}'
          'and epoch {ec.gan_epoch_num}\n')

    print(f'\nDataset and split of real data for evaluation are:'
            '{ec.dataset} {ec.split}\n')

    print(f'Metrics selected for evaluation: {ec.metrics}\n')

    if not (os.path.exists(ec.save_evaluations_folder)):
        os.makedirs(ec.save_evaluations_folder)

    for metric in ec.metrics:

        if metric == 'FID':
            # create numpy array to be loaded for computing prd
            print("Reading real features into a numpy array\n")
            real_features = eval_ut.squash_features(ec.data_real_features)

            print("Reading generated features into a numpy array\n")
            gen_features = eval_ut.squash_features(ec.data_gen_features)
            gen_features = np.random.permutation(gen_features)

            print(real_features.shape)
            print(gen_features.shape)

            print('Calculating FID score between real and generated features'
                  'of trial {} epoch {}: '.format(ec.gan_trial_num,
                                                  ec.gan_epoch_num))
            fid = eval_ut.calculate_fid(real_features, gen_features)

            print(fid)
            ec.save_columns.append(fid)

        elif metric == 'PRD':
            # create numpy array to be loaded for computing prd
            print("Reading real features into a numpy array\n")
            real_features = eval_ut.squash_features(ec.data_real_features)

            print("Reading generated features into a numpy array\n")
            gen_features = eval_ut.squash_features(ec.data_gen_features)
            
            print("Calculating prd from embeddings\n")
            prd_data_1 = prd.compute_prd_from_embedding(real_features,
                                                        gen_features)
            np.save(ec.save_evaluations_folder + 'prd_trial_'
                    + str(ec.gan_trial_num) + '_epoch_' + str(ec.gan_epoch_num)
                    + '.npy', prd_data_1)

            print('Plotting prd curve of GAN 1\n')
            prd.plot([prd_data_1[:2]], ['GAN_trial_' + str(ec.gan_trial_num)
                                        + '_epoch_' + str(ec.gan_epoch_num)],
                     ec.save_evaluations_folder + 'gan1_trial_'
                     + str(ec.gan_trial_num) + '_epoch_'
                     + str(ec.gan_epoch_num) + '.jpg')

        else:
            print('\nInvalid metric\n')

    print("Configuration for the run: \n")
    print(dict(zip(ec.save_column_names, ec.save_columns)))
    ut.save_config(ec.save_evaluations, ec.save_column_names, ec.save_columns)


if __name__ == "__main__":
    main()
