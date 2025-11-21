import os
import numpy as np
from torch.utils.data import Dataset


class IRDS_Dataset(Dataset):

    def __init__(self, dataset_root=None, movement='m01', is_train=True):

        super().__init__()
        self.dataset_root = os.path.join(dataset_root, 'data', 'IRDS')
        '''
        0 - spine base
        1 - spine mid
        2 - neck
        3 - head
        4 - shoulder left
        5 - elbow left
        6 - wrist left
        7 - hand left
        8 - shoulder right
        9 - elbow right
        10 - wrist right
        11 - hand right
        12 - hip left
        13 - knee left
        14 - ankle left
        15 - foot left
        16 - hip right
        17 - knee right
        18 - ankle right
        19 - foot right
        20 - spine shoulder
        21 - hand tip left
        22 - tumb left
        23 - hand tip right
        24 - tumb right
        '''
        self.connectivity = [(0, 1), (0, 16), (0, 12), (1, 20), (20, 2),
                             (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (6, 22),
                             (7, 21), (20, 8), (8, 9), (9, 10), (10, 11),
                             (10, 24), (11, 23), (12, 13), (13, 14), (14, 15),
                             (16, 17), (17, 18), (18, 19)]
        self.root_id = 0
        if movement != 'all':
            self.movement_id = int(movement[1:]) - 1
        else:
            self.movement_id = -1
        self.movement = movement
        self.dataset_fps = 30
        '''
        m01 - Elbow flexion left
        m02 - Elbow flexion right
        m03 - Shoulder flexion left
        m04 - Shoulder flexion right
        m05 - Shoulder abduction left
        m06 - Shoulder abduction right
        m07 - Shoulder forward elevation
        m08 - Side tap left
        m09 - Side tap right
        '''
        if is_train:
            res_ids = [0, 1, 2]
        else:
            res_ids = [3]

        self.samples = []
        self.labels = []
        self.cls_num = 2  # correct or incorrect
        self.num_per_cls_dict = {
            0: 0,  # incorrect
            1: 0   # correct
        }
        correct_count = 0
        incorrect_count = 0

        idx = 0
        for filename in sorted(
                os.listdir(os.path.join(self.dataset_root, 'Simplified'))):
            idx += 1
            if idx % 4 not in res_ids:
                continue

            SubjectID = int(filename.split('_')[0])
            DateID = int(filename.split('_')[1])
            GestureLabel = int(filename.split('_')[2])
            if self.movement != 'all':
                if GestureLabel != self.movement_id:
                    continue

            RepetitionNo = int(filename.split('_')[3])
            '''
            1 - correct gesture
            2 - incorrect gesture
            3 - incorrect but poorly executed
            '''
            CorrectLabel = int(filename.split('_')[4])
            if CorrectLabel == 1:
                correct_count = correct_count + 1
            else:
                incorrect_count = incorrect_count + 1
            Position = filename.split('_')[5]
            sample = np.loadtxt(os.path.join(self.dataset_root, 'Simplified',
                                             filename),
                                delimiter=',')
            sample = sample.reshape(sample.shape[0], -1, 3)
            # for unified training with multi-calss
            class_label = 2 * GestureLabel + (CorrectLabel == 1)
            self.samples.append(sample)
            self.labels.append((SubjectID, DateID, GestureLabel, RepetitionNo,
                                CorrectLabel, Position, class_label))
            # CorrectLabel is a bool data
            self.num_per_cls_dict[CorrectLabel == 1] += 1

    def get_cls_num_statistic(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rotmat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        keypoints = np.matmul(self.samples[index], rotmat)
        if self.movement == 'all':
            return keypoints, self.labels[index][2], self.labels[index][4] == 1
        else:
            return keypoints, self.labels[index][2], self.labels[index][4] == 1
