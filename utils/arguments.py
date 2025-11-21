import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--root',
                        type=str,
                        default='./',
                        help='path to root of this project')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--vis',
                        action='store_true',
                        help='visualize movement CAM')

    # dataset and preprocess
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['IRDS', 'UIPRMD_Kinect', 'UIPRMD_Vicon', 'PushUp', 'REHAB24-6'],
        help='dataset type')
    parser.add_argument('--Pnorm',
                        action='store_true',
                        help='normalize orientation of skeletons')
    parser.add_argument('--aug_angle',
                        type=int,
                        nargs='+',
                        default=0,
                        help='rotation augmentation for PushUp dataset')

    # classification model
    parser.add_argument('--model',
                        type=str,
                        default='ri-gcn',
                        choices=['gcn', 'ri-gcn', 'va-gcn', 'msst-gcn', 'gcn-scl'],
                        help='type of model to be used')
    parser.add_argument('--strategy',
                        type=str,
                        choices=['uniform', 'distance'],
                        default='uniform',
                        help='strategy of constructing A matrix in ST-GCN')

    # training hyper parameters
    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        help='epochs to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size of data')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--unified',
                        action='store_true',
                        help='use all movement of a dataset to train a unified model')
    parser.add_argument('--type_test',
                        action='store_true',
                        help='type-specific testing for a unified model')
    parser.add_argument('--tmax',
                        type=float,
                        default=0.9,
                        help='max threshold for scl method in type-specific testing')
    parser.add_argument('--tmin',
                        type=float,
                        default=0.3,
                        help='min threshold for scl method in type-specific testing')
    parser.add_argument('--num_classes', type=int, default=18, help='total classes for unified model')
    parser.add_argument('--inferdir',
                        type=str,
                        default='',
                        help='The folder used for inference, which is the path to save the trained model during training')
    parser.add_argument('--gamma', default=0.3, type=float, help='CDT hyperparameter')
    parser.add_argument('--tau', default=1.0, type=float, help='LA hyperparameter')
    parser.add_argument('--time_mix',
                        action='store_true',
                        help='use time mix-up to augment data')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0005,
                        help='use time mix-up to augment data')
    parser.add_argument('--loss',
                        type=str,
                        default='ce',
                        choices=['ce', 'vs', 'joint_ce', 'joint_vs', 'fcl', 'joint_ce_fcl', 'asyfcl', 'mfcl', 'lsce', 'jlsce', 'fcls', 'scl'],
                        help='type of loss to be used')
    parser.add_argument('--lam',
                        type=float,
                        default=1.0,
                        help='weight of joint learning')
    parser.add_argument('--temp',
                        type=float,
                        default=0.1,
                        help='weight of joint learning')
    parser.add_argument('--ga',
                        type=float,
                        default=2.0,
                        help='gamma for focal loss')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.75,
                        help='alpha for focal loss')
    parser.add_argument('--margin',
                        type=float,
                        default=0.1,
                        help='margin for margin focal loss')
    parser.add_argument('--is_ri',
                        action='store_true',
                        help='use rotation invariant descriptor'

    )
    parser.add_argument('--drop_out',
                        type=float,
                        default=0,
                        help='drop out probability')
    parser.add_argument('--dummy_class',
                        type=int,
                        default=0,
                        help='use dummy class to train model'
    )
    parser.add_argument('--amp',
                        action='store_true',
                        help='use amp to train model')
    parser.add_argument('--eps',
                        type=float,
                        default=0.1,
                        help='eps for label smooth')
    parser.add_argument('--theta',
                        type=float,
                        default=0.5,
                        help='threshold of the binary classification in the SCL-HSN method'
                        )

    args = parser.parse_args()
    return args
