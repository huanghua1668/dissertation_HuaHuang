# Dataset, code, and results of Hua's dissertation

## Dataset
Data extraction process:
1. Use extract_lane_change.py to extract lane change sequences from both dataset, and extract_lane_change_merge_after.py to extract lane change abortions.
2. Use extract_sample.py and extract_sample_merge_after.py to extract the exact snapshot from the lane change sequences for the downstream prediction task.
3. Use data_preprocess.py to assemble the extracted samples; split to training and testing; generate OOD samples.

 Dataset | Description | Generating program | 
----|----|----
 us80.npz| Raw lane changes from US80, each row is one sample, with record [merge in front/after(0/1), u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y].| extract_sample.py, extract_sample_merge_after.py, and preprocess_both_dataset() in data_preprocess.py
 us101.npz| Raw lane changes from US101, each row is one sample, with record [merge in front/after(0/1), u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]| extract_sample.py, extract_sample_merge_after.py, and preprocess_both_dataset() in data_preprocess.py|
samples_relabeled_by_decrease_in_dx.npz| Samples with label based on $\Delta x$, each row is a sample with record [index, dt, u0, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2, y]| extract_samples_relabel_as_change_in_distance() in extract_sample.py |
combined_dataset.npz| Combine us80 and us101 datasets, do feature selection, and normalize them to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test, f['e']=xGenerated), each sample is of feature x = [dv0, dv1, dx0, dx1] | prepare_validate_and_generate_ood() in data_process.py |
combined_dataset_before_feature_selection.npz| Combine us80 and us101 datasets and normalize them to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test), each sample is of feature x = [v_ego, dv0, dv1, dv2, dx0, dx1, dx2, dy0, dy1, dy2]| prepare_validate_and_feature_selection() in data_process.py|
combined_dataset_trainUs80_testUs101.npz| us80 as training dataset, us101 as testing dataset. Normalize samples to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test, f['e']=xGenerated), each sample is of feature x = [dv0, dv1, dx0, dx1] | prepare_validate_and_generate_ood_trainUs80_testUs101() in data_process.py|
combined_dataset_trainUs101_testUs80.npz| us101 as training dataset, us80 as testing dataset. Normalize samples to have 0 mean and 1 std, in which f['a']=x_train, f['b'']=y_train, f['c']=x_test, f['d']=y_test, f['e']=xGenerated), each sample is of feature x = [dv0, dv1, dx0, dx1] | prepare_validate_and_generate_ood_trainUs80_testUs101() in data_process.py|
lane_changes_trajectories.csv| demonstrative lane change trajectories | extract_lane_changes.py |
min_dis_within_us80.npz| percentage of minimum distance to other samples in dataset us80 | cal_distance() in data_preprocess.py|
min_dis_within_us101.npz| percentage of minimum distance to other samples in dataset us101 | cal_distance() in data_preprocess.py|

# Results
 Figure/table| Supporting data| Generating program| Plot function (in plot_utils.py)
----|----|----|----
 Figure 3.2: Lane changes| lane_changes_trajectories.csv|extract_lane_changes.py | plot_trajectory()|
 Figure 3.3: Label based on the change in ∆x.| samples_relabeled_by_decrease_in_dx.npz| extract_samples_relabel_as_change_in_distance() in extract_sample.py | visualize_sample_labeled_by_dx()|
 Figure 3.4: Accelerations of lag vehicles.| lane_changes_trajectories.csv| extract_lane_changes.py| plot_accelerations()|
 Figure 3.5: Lane changes in dataset I-80. | us80.npz | preprocess_both_dataset() in data_preprocess.py | plot_ngsim_scatter()| 
 Figure 3.6: Lane changes in dataset US-101. | us101.npz | preprocess_both_dataset() in data_preprocess.py| plot_ngsim_scatter()| 
 Figure 4.1: Confidence map for α = 0 (left) and α = 1 (right).| moons_confidence_alpha0.npz, moons_confidence_alpha1.npz|moons_csnn_alpha_effect_demo.py| plot_confidence_map_alpha_impact()|
 Figure 4.2: Evolution of the support circles during training.| /evolvement|moons_csnn_evolvement.py| plot_evolvement() |
 Figure 4.3: Evolution of the confidence during training.| /evolvement|moons_csnn_evolvement.py| plot_evolvement()|
 Figure 4.4: The radius penalty effect. | /redius_penalty_effect|moons_csnn_radius_penalty_effect.py|plot_radius_penalty_effect()
 Figure 4.5: Samples in the moons dataset. |moons_train_test_ood.npz| moons_csnn.py|plot_moons_scatter()
 Figure 4.6: Convergence of test accuracy, non-zero output ratio, and AUROC.|moons_acc_nzs_auroc.npz| moons_csnn.py|plot_save_acc_nzs_auroc()|
 Figure 4.7: Histograms of confidences for in-distribution and OOD samples.|moons_hist_confidence.npz|moons_csnn.py|plot_distribution()|
 Figure 4.8: ROC curve of the moons dataset|moons_roc.npz|moons_csnn.py|plot_save_roc()|
 Figure 4.9: Best test set accuracy vs. number of selected features| |ngsim_feature_selection.py|plot_feautre_selection_results()
 Figure 4.10: Minimum distance to other in-distribution samples.|min_dis_within_us80.npz, min_dis_within_us101.npz|cal_distance() in data_preprocess.py|plot_min_distance_within_dataset()
 Figure 4.11: Histograms of confidences on the NGSIM dataset and the generated OOD samples.|ngsim_hist_confidence_epoch240.npz|ngsim_csnn.py|plot_distribution()|
 Figure 4.12: OOD detection ROC curve for one run of training the CSNN on the NGSIM dataset.|ngsim_roc_epoch240.npz|ngsim_csnn.py|plot_save_roc()|
 Figure 4.13: Test accuracy and AUROC vs α for the CSNN.|/ngsim_lambda_effect|ngsim_csnn.py|plot_auc_acc_csnn()|
 Table 4.2: Test accuracy and AUROC in OOD detection, averaged of 10 independent runs. (MLP)|mlp_mean_std_accs_aucs_net4.npz|ngsim_mlp.py| |
 Table 4.2: Test accuracy and AUROC in OOD detection, averaged of 10 independent runs. (DUQ)|/duq|ngsim_duq.py| |
 Table 4.2: Test accuracy and AUROC in OOD detection, averaged of 10 independent runs. (Deep ensemble)|/deep_ensemble|ngsim_deep_ensemble.py| |
 Figure 4.14: Entropy distribution of in-distribution and OOD samples.|/deep_ensemble/ngsim_hist_confidence.npz| ngsim_deep_ensemble.py| plot_distribution()|
 Figure 4.15: ROC curve obtained with deep ensemble.|/deep_ensemble/ngsim_roc.npz| ngsim_deep_ensemble.py| plot_save_roc()|
 Figure 4.16: Entropy of average prediction.|moons_confidence_map_deep_ensemble.npz| moons_deep_ensemble.py| plot_moons_de()|
 Table 4.3: Generalization of network with compact support.|/train_us80_test_us101 and /train_us101_test_us80| ngsim_mlp.py, ngsim_csnn.py, ngsim_duq.py, ngsim_deep_ensemble.py|  |
 
 
 
 
 
 
 
