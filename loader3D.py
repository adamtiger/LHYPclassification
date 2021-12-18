from heartcontour.models.lvhyp.transformer import Sample
from heartcontour.models.lvhyp.loader import DataType
from heartcontour.data_wrangling.lvhyp_reader import Pathology
from heartcontour.utils import progress_bar
from heartcontour.utils import get_logger
from enum import Enum, auto
import numpy as np
import torch
import json
import os

logger = get_logger(__name__)

# Two datasets are necessary:
# LA dataset for LA model training
# SA dataset for SA model training


class LhypDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_paths,  # path to the sample pickles to read them in
        dtypes,        # DataType, list
        preproc,       # converts to PIL image, crop to 190x190 and resizes 140x140
        augment,       # does the augmentation on the PIL image then converts to tensor
        device,        # GPU or CPU
        mode = 'NORM'  # HYP vs NORM or HCM vs. others
    ):
        self.sample_paths = sample_paths
        self.dtypes = dtypes
        self.preproc = preproc
        self.augment = augment
        self.device = device
        self.mode = mode
        # read data to memory
        self.images, self.labels, self.patient_ids = self.read_samples_to_ram()
        print(len(self.labels), np.sum(self.labels))

    def read_samples_to_ram(self):
        # image storage
        images = dict()
        for dtype in self.dtypes:
            images[dtype] = list()
        # label storage
        labels = list()
        # patient id storage
        patient_ids = list()

        def intensity_rescale(image, thresholds=(1.0, 99.0)):
            val_l, val_h = np.percentile(image, thresholds)
            image[image < val_l] = val_l
            image[image > val_h] = val_h
            image = (image - val_l) / (val_h - val_l + 1e-5)
            image = image.astype(np.float32)
            return image

        def transform_and_save(dtype, imgs, length, path):
            # imgs: list of ndarrays
            imgs = [intensity_rescale(im) for im in imgs[0:length]]
            imgs = self.preproc(imgs)
            images[dtype].append(imgs)

        for cntr, smp_path in enumerate(self.sample_paths, 1):
            valid = True
            smp = Sample.deserialize(smp_path)
            if self.mode == 'HCM':
                valid = (smp.pathology != Pathology.NORMAL)
            exl = 12  # expected length of image vectors
            # first filter the necessary image
            if DataType.SA in self.dtypes:  # if we need sa images
                if len(smp.sa_bas) >= exl and len(smp.sa_mid) >= exl and len(smp.sa_api) >= exl:
                    imgs = list()
                    for k in range(exl // 2):
                        imgs.append(smp.sa_bas[k * 2])
                        imgs.append(smp.sa_mid[k * 2])
                        imgs.append(smp.sa_api[k * 2])
                    transform_and_save(DataType.SA, imgs, 3 * exl, smp_path)
                else:
                    valid = False
            if valid and DataType.LA2CH in self.dtypes:
                if len(smp.la_2ch) >= exl:
                    transform_and_save(DataType.LA2CH, smp.la_2ch, exl, smp_path)
                else:
                    valid = False
            if valid and DataType.LA4CH in self.dtypes:
                if len(smp.la_4ch) >= exl:
                    transform_and_save(DataType.LA4CH, smp.la_4ch, exl, smp_path)
                else:
                    valid = False
            if valid and DataType.LALVOT in self.dtypes:
                if len(smp.la_lvot) >= exl:
                    transform_and_save(DataType.LALVOT, smp.la_lvot, exl, smp_path)
                else:
                    valid = False
            if valid and DataType.LALE in self.dtypes:
                if len(smp.lale) >= 3:
                    transform_and_save(DataType.LALE, smp.lale, 3, smp_path)
                else:
                    valid = False
            if valid and DataType.SALE in self.dtypes:
                if len(smp.sale) >= 6 and not(True in (smp is None for smp in smp.sale)):
                    transform_and_save(DataType.SALE, smp.sale, 6, smp_path)
                else:
                    valid = False
            # choose the label (0 - normal, 1 - HCM)
            if valid:
                if self.mode == 'NORM':
                    patient_ids.append(smp.patient_id)
                    if smp.pathology == Pathology.NORMAL:
                        labels.append(0)
                    else:
                        labels.append(1)
                else:
                    patient_ids.append(smp.patient_id)
                    if smp.pathology == Pathology.HCM:
                        labels.append(0)
                    else:
                        labels.append(1)
            progress_bar(cntr, len(self.sample_paths), 10)
        return images, labels, patient_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = dict()
        label = self.labels[index]
        patientid = self.patient_ids[index]
        for dt in self.dtypes:
            img = self.augment(self.images[dt][index])  # stacks together several images
            img = np.stack(img, axis=2)
            sample[dt] = torch.from_numpy(img).unsqueeze(0).type(torch.float).to(self.device)
        sample['label'] = torch.tensor(label, dtype=torch.long, device=self.device)
        sample['pid'] = patientid
        return sample

    @staticmethod
    def split_data(sample_folder_path, num_cross_val_divides, train_ratio=0.7, validation_ratio=0.15):
        sample_names = [sn for sn in os.listdir(sample_folder_path)]
        np.random.shuffle(sample_names)
        paths = list()
        for name in sample_names:
            p = os.path.join(sample_folder_path, name)
            paths.append(p)
        train_split_idx = int(len(paths) * train_ratio)
        validation_split_idx = int(len(paths) * (train_ratio + validation_ratio))
        splitting = dict()
        for cvi in range(num_cross_val_divides):
            splitting[str(cvi)] = dict()
            cross_val_paths = paths[0:validation_split_idx]
            np.random.shuffle(cross_val_paths)
            paths_train = cross_val_paths[0:train_split_idx]
            paths_val = cross_val_paths[train_split_idx:validation_split_idx]
            splitting[str(cvi)]["train"] = paths_train
            splitting[str(cvi)]["valid"] = paths_val
        paths_test = paths[validation_split_idx:]
        np.random.shuffle(paths_test)
        splitting["test"] = paths_test
        return splitting
    
    @staticmethod
    def split_data2(sample_folder_path, reference_file_path, num_cross_val_divides, train_ratio=0.7):
        paths_test = None
        sample_test_names = None
        with open(reference_file_path, 'rt') as js:
            refsplitting = json.load(js)
            paths_test = refsplitting["test"]
            sample_test_names = set([pt.split(os.sep)[-1] for pt in paths_test])
        sample_names = [sn for sn in os.listdir(sample_folder_path) if not(sn in sample_test_names)]
        np.random.shuffle(sample_names)
        paths = list()
        for name in sample_names:
            p = os.path.join(sample_folder_path, name)
            paths.append(p)
        train_split_idx = int(len(paths) * train_ratio)
        splitting = dict()
        for cvi in range(num_cross_val_divides):
            splitting[str(cvi)] = dict()
            np.random.shuffle(paths)
            paths_train = paths[0:train_split_idx]
            paths_val = paths[train_split_idx:]
            splitting[str(cvi)]["train"] = paths_train
            splitting[str(cvi)]["valid"] = paths_val
        np.random.shuffle(paths_test)
        splitting["test"] = paths_test
        return splitting
