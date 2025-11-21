import datetime
import os
import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn import metrics
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from mvn.dataset import AdaptiveMoveDataset

from utils.arguments import get_args
from utils.setup import setup_experiment, setup_seed


def one_move(movement, exp_dir, result_dict, device, logger, args, aug_angle, test_movement=None):
    movement_dir = os.path.join(exp_dir, movement)
    os.makedirs(movement_dir, exist_ok=True)

    if movement.startswith('IRDS'):
        num_joints = 25
    elif movement.startswith('UIPRMD_Kinect'):
        num_joints = 22
    elif movement.startswith('UIPRMD_Vicon'):
        num_joints = 39
    elif movement.startswith('REHAB24-6'):
        num_joints = 26
    else:
        num_joints = None

    # test_movement is None:
    # 1.type-specific training and testing
    # 2.unified training and unified testing
    # test_movement is not None:
    # 1.unified training and type-specific testing
    movement_class = movement if test_movement is None else test_movement

    valid_dataset = AdaptiveMoveDataset(dataset_root=args.root,
                                        movement_class=movement_class,
                                        norm_orient=args.Pnorm,
                                        aug_angle=aug_angle,
                                        is_train=False)
    valid_dataloader = DataLoader(valid_dataset,
                                  args.batch_size,
                                  shuffle=False,
                                  num_workers=1)


    import dill
    model_path = f"./{args.dataset}_model_full.pt"
    model = torch.load(model_path, map_location=device, pickle_module=dill)
    model.to(device)

    model.eval()
    labels_gt = []
    labels_pred = []
    all_scores = []

    if args.model == 'gcn-scl':
        memory_correct_features = []
        for _, (samples_batch, _, labels_batch) in enumerate(valid_dataloader):
            correct_sub_batch = samples_batch[labels_batch != 0].float().to(device)
            with torch.no_grad():
                correct_feature_batch = model(correct_sub_batch)
                correct_feature_batch = F.normalize(correct_feature_batch, p=2, dim=1)
                memory_correct_features.append(correct_feature_batch)

        with torch.no_grad():
            memory_correct_features = torch.cat(memory_correct_features, dim=0)
            sum_features = memory_correct_features.sum(dim=0)
            var_features = memory_correct_features.var(dim=0)
            inverse_variance = 1 / (var_features + 1e-8)

            l1_norm = inverse_variance.abs().sum()
            normalized_weights = inverse_variance / l1_norm

            ref_features = normalized_weights * sum_features
            ref_features = F.normalize(ref_features, p=2, dim=0)

    for j, (samples_batch, _, labels_batch) in enumerate(valid_dataloader):
        samples_batch = samples_batch.float().to(device)
        labels_batch_binary = labels_batch != 0

        with torch.no_grad():
            label_pred = model(samples_batch)

            if args.model == 'msst-gcn':
                label_pred = sum(label_pred)
            # remove dummy class
            if args.dummy_class > 0:
                label_pred = label_pred[:, 0:args.num_classes]

        if args.model == 'gcn-scl':
            label_pred = F.normalize(label_pred, p=2, dim=1)
            scores = F.cosine_similarity(label_pred, ref_features.unsqueeze(0), dim=1)
            label_pred = (scores >= args.theta).float()

            all_scores.append((1 + scores.detach().cpu().numpy()) / 2.0)  # 1.normalize to [0,1]
        else:
            probs = F.softmax(label_pred, dim=1)
            scores = probs[:, 1]
            all_scores.append(scores.cpu().numpy())
            label_pred = torch.argmax(label_pred, dim=1)

        labels_gt.append(np.array(labels_batch_binary))
        labels_pred.append(np.array(label_pred.cpu()))

    labels_gt = np.concatenate(labels_gt, axis=0)
    labels_pred = np.concatenate(labels_pred, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    accuracy = metrics.accuracy_score(y_true=labels_gt, y_pred=labels_pred)
    recall = metrics.recall_score(y_true=labels_gt, y_pred=labels_pred)
    precision = metrics.precision_score(y_true=labels_gt,
                                        y_pred=labels_pred,
                                        zero_division=0)

    accuracy_invert = metrics.accuracy_score(y_true=1 - labels_gt, y_pred=1 - labels_pred)
    recall_invert = metrics.recall_score(y_true=1 - labels_gt,
                                         y_pred=1 - labels_pred)
    precision_invert = metrics.precision_score(y_true=1 - labels_gt,
                                               y_pred=1 - labels_pred,
                                               zero_division=0)
    f1_score = metrics.f1_score(y_true=1 - labels_gt, y_pred=1 - labels_pred)

    # ===================== metric =====================
    # 1.AUROC, 1 - incorrect, 0 - correct
    roc_auc = metrics.roc_auc_score(1 - labels_gt, 1 - all_scores)

    # 2.AUPRC, 1 - incorrect, 0 - correct
    pr_auc = metrics.average_precision_score(1 - labels_gt, 1 - all_scores)

    # 1. ROC curve
    fpr, tpr, _ = metrics.roc_curve(1 - labels_gt, 1 - all_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_path = os.path.join(movement_dir, f"ROC.pdf")
    plt.savefig(roc_path, format='pdf')
    plt.close()

    # 2. PR curve
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(1 - labels_gt, 1 - all_scores)
    plt.figure()
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    pr_path = os.path.join(movement_dir, f"PR.pdf")
    plt.savefig(pr_path, format='pdf')
    plt.close()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(
        '{} | {}: Incorrect for 1: accuracy {:.4f}, recall {:.4f}, precision {:.4f}, f1_score {:.4f}, roc_auc {:.4f}, pr_auc {:.4f}'.
            format(current_time, movement_class, accuracy_invert, recall_invert, precision_invert, f1_score, roc_auc, pr_auc))

    result_dict[movement_class]['accuracy'] = float(accuracy)
    result_dict[movement_class]['recall'] = float(recall)
    result_dict[movement_class]['precision'] = float(precision)
    result_dict[movement_class]['accuracy_invert'] = float(accuracy_invert)
    result_dict[movement_class]['recall_invert'] = float(recall_invert)
    result_dict[movement_class]['precision_invert'] = float(precision_invert)
    result_dict[movement_class]['f1_score'] = float(f1_score)
    result_dict[movement_class]['roc_auc'] = float(roc_auc)
    result_dict[movement_class]['pr_auc'] = float(pr_auc)


def inference(args):
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    exp_dir, logger = setup_experiment(args, is_inference=True)
    logger.info(args)

    for aug_angle in args.aug_angle if isinstance(args.aug_angle,
                                                  list) else [args.aug_angle]:

        result_dict = defaultdict(dict)
        movement_list = {
            'IRDS': [
                'IRDS_m01', 'IRDS_m02', 'IRDS_m03', 'IRDS_m04', 'IRDS_m05',
                'IRDS_m06', 'IRDS_m07', 'IRDS_m08', 'IRDS_m09'
            ],
            'UIPRMD_Kinect': [
                'UIPRMD_Kinect_m01', 'UIPRMD_Kinect_m02', 'UIPRMD_Kinect_m03',
                'UIPRMD_Kinect_m04', 'UIPRMD_Kinect_m05', 'UIPRMD_Kinect_m06',
                'UIPRMD_Kinect_m07', 'UIPRMD_Kinect_m08', 'UIPRMD_Kinect_m09',
                'UIPRMD_Kinect_m10'
            ],
            'UIPRMD_Vicon': [
                'UIPRMD_Vicon_m01', 'UIPRMD_Vicon_m02', 'UIPRMD_Vicon_m03',
                'UIPRMD_Vicon_m04', 'UIPRMD_Vicon_m05', 'UIPRMD_Vicon_m06',
                'UIPRMD_Vicon_m07', 'UIPRMD_Vicon_m08', 'UIPRMD_Vicon_m09',
                'UIPRMD_Vicon_m10'
            ],
            'REHAB24-6': [
                'REHAB24-6_m01', 'REHAB24-6_m02', 'REHAB24-6_m03',
                'REHAB24-6_m04', 'REHAB24-6_m05', 'REHAB24-6_m06'
            ]
        }[args.dataset]
        if args.unified:
            movement = args.dataset + '_all'
            if args.type_test:
                for test_movement in movement_list:
                    one_move(movement, exp_dir, result_dict, device,
                                           logger, args, aug_angle, test_movement=test_movement)
            else:
                one_move(movement, exp_dir, result_dict, device,
                                       logger, args, aug_angle)
        else:
            for movement in movement_list:
                one_move(movement, exp_dir, result_dict, device,
                                       logger, args, aug_angle)

        accuracy_list = []
        recall_list = []
        precision_list = []
        accuracy_list_invert = []
        recall_list_invert = []
        precision_list_invert = []
        f1_score_list = []
        roc_auc_list = []
        pr_auc_list = []
        for move in result_dict.keys():
            accuracy_list.append(result_dict[move]['accuracy'])
            recall_list.append(result_dict[move]['recall'])
            precision_list.append(result_dict[move]['precision'])
            accuracy_list_invert.append(result_dict[move]['accuracy_invert'])
            recall_list_invert.append(result_dict[move]['recall_invert'])
            precision_list_invert.append(result_dict[move]['precision_invert'])
            f1_score_list.append(result_dict[move]['f1_score'])
            roc_auc_list.append(result_dict[move]['roc_auc'])
            pr_auc_list.append(result_dict[move]['pr_auc'])

        result_dict['mean']['accuracy'] = float(np.mean(accuracy_list))
        result_dict['mean']['recall'] = float(np.mean(recall_list))
        result_dict['mean']['precision'] = float(np.mean(precision_list))
        result_dict['mean']['accuracy_invert'] = float(np.mean(accuracy_list_invert))
        result_dict['mean']['recall_invert'] = float(np.mean(recall_list_invert))
        result_dict['mean']['precision_invert'] = float(np.mean(precision_list_invert))
        result_dict['mean']['f1_score'] = float(np.mean(f1_score_list))
        result_dict['mean']['roc_auc'] = float(np.mean(roc_auc_list))
        result_dict['mean']['pr_auc'] = float(np.mean(pr_auc_list))
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(
            '{} | Incorrect for 1: mean accuracy {}, mean recall {}, mean precision {}, mean f1_score {}, mean roc_auc {}, mean pr_auc {}'.format(
                current_time, result_dict['mean']['accuracy_invert'], result_dict['mean']['recall_invert'], result_dict['mean']['precision_invert'], result_dict['mean']['f1_score'],
                result_dict['mean']['roc_auc'], result_dict['mean']['pr_auc']))

        with open(os.path.join(exp_dir, 'result_inference.yaml'), 'w+',
                  encoding='utf-8') as f:
            yaml.dump(dict(result_dict), f, allow_unicode=True, sort_keys=False)


if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)
    inference(args)
