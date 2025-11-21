import os
import numpy as np
from torch.utils.data import Dataset


class REHAB24_6_Dataset(Dataset):

    def __init__(self, dataset_root=None, movement='m01', is_train=True):
        super().__init__()
        self.dataset_root = os.path.join(dataset_root, 'data', 'REHAB24-6')
        self.is_train = is_train
        self.connectivity = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),  # trunk
            (3, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),  # left arm
            (3, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),  # right arm
            (0, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # left leg
            (0, 21),
            (21, 22),
            (22, 23),
            (23, 24),
            (24, 25)  # right leg
        ]


        self.root_id = 0
        self.dataset_fps = 30
        '''
        ref: REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods. https://doi.org/10.1007/978-3-031-75823-2_2
        
        m01 - Arm abduction: sideway raising of the straightened right arm
        m02 - Arm VW: fluent transition of arms between V (arms straight up) and W (elbows down, hands up) shape
        m03 - Push-ups: push-ups with hands on a table
        m04 - Leg abduction: sideway raising of the straightened leg
        m05 - Leg lunge: pushing a knee of the back leg down while keeping a right angle on the front knee
        m06 - Squats
        '''

        if is_train:
            res_ids = [0, 1, 2]
        else:
            res_ids = [3]

        self.samples = []
        self.labels = []
        correct_count = 0
        incorrect_count = 0

        idx = 0
        for filename in sorted(
                os.listdir(
                    os.path.join(self.dataset_root, '3d_joints_segmented'))):
            if movement != 'all':
                if movement not in filename:
                    continue
            idx = idx + 1
            if idx % 4 not in res_ids:
                continue

            filepath = os.path.join(
                os.path.join(self.dataset_root, '3d_joints_segmented', filename))
            keypoints = np.load(filepath)
            keypoints = keypoints[:, :, 0:3]  # remove the redundant data
            keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
            GestureLabel = int(filename.split('-')[0][1:3])
            CorrectLabel = int(filename.split('-')[5][0])
            self.samples.append(keypoints)
            self.labels.append((GestureLabel, CorrectLabel))
            if CorrectLabel == 1:
                correct_count = correct_count + 1
            else:
                incorrect_count = incorrect_count + 1


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index] / 100.
        label_cls = self.labels[index][0]
        label = self.labels[index][1]

        return sample, label_cls, label

