# adapted from: https://github.com/thuml/Anomaly-Transformer/data_factory/data_loader.py
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from datasets.util import downsample


class TSDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.dataset_name = cfg.DATA.NAME
        self.data_dir = os.path.join(cfg.DATA.BASE_DIR, cfg.DATA.NAME)
        self.win_size = cfg.DATA.WIN_SIZE
        self.step = cfg.DATA.TRAIN_STEP if split == "train" else cfg.DATA.TEST_STEP
        self.scale = cfg.DATA.SCALE
        self.split = split
        self.train_ratio = cfg.DATA.TRAIN_RATIO
        self.downsample_rate = cfg.DATA.DOWNSAMPLE_RATE

        self.train, self.val, self.test, self.train_labels, self.val_labels, self.test_labels = self._load_data()
        self.test_labels = np.squeeze(self.test_labels)
        self._normalize_data()

    def _save_downsampled_data(self, train, val, test, train_labels, val_labels, test_labels):
        assert self.downsample_rate > 1
        np.save(os.path.join(self.data_dir, f"downsampled_train_rate{self.downsample_rate}.npy"), train)
        np.save(os.path.join(self.data_dir, f"downsampled_val_rate{self.downsample_rate}.npy"), val)
        np.save(os.path.join(self.data_dir, f"downsampled_test_rate{self.downsample_rate}.npy"), test)
        np.save(os.path.join(self.data_dir, f"downsampled_train_labels_rate{self.downsample_rate}.npy"), train_labels)
        np.save(os.path.join(self.data_dir, f"downsampled_val_labels_rate{self.downsample_rate}.npy"), val_labels)
        np.save(os.path.join(self.data_dir, f"downsampled_test_labels_rate{self.downsample_rate}.npy"), test_labels)
    
    def _load_downsampled_data(self):
        train = np.load(os.path.join(self.data_dir, f"downsampled_train_rate{self.downsample_rate}.npy"))
        val = np.load(os.path.join(self.data_dir, f"downsampled_val_rate{self.downsample_rate}.npy"))
        test = np.load(os.path.join(self.data_dir, f"downsampled_test_rate{self.downsample_rate}.npy"))
        train_labels = np.load(os.path.join(self.data_dir, f"downsampled_train_labels_rate{self.downsample_rate}.npy"))
        val_labels = np.load(os.path.join(self.data_dir, f"downsampled_val_labels_rate{self.downsample_rate}.npy"))
        test_labels = np.load(os.path.join(self.data_dir, f"downsampled_test_labels_rate{self.downsample_rate}.npy"))
        return train, val, test, train_labels, val_labels, test_labels

    def _load_data(self):
        raise NotImplementedError

    def _split_train_val(self, train, train_labels):
        # use the latter portion of the training data as validation data.
        assert self.train_ratio < 1.0

        train_len = int(len(train) * self.train_ratio)
        val = train[train_len:].copy()
        val_labels = train_labels[train_len:].copy()
        train = train[:train_len]
        train_labels = train_labels[:train_len]
        
        return train, val, train_labels, val_labels

    def _normalize_data(self):
        if self.scale in ("standard", "instance"):
            scaler = StandardScaler()
        elif self.scale == "min-max":
            scaler = MinMaxScaler()
        elif self.scale == "none":
            return
        else:
            raise ValueError

        self.train = scaler.fit_transform(self.train)
        self.val = scaler.transform(self.val)
        self.test = scaler.transform(self.test)

    def print_data_stats(self):
        if self.split == 'train':
            print(f"Train data shape: {self.train.shape}, mean: {np.mean(self.train, axis=0)}, std: {np.std(self.train, axis=0)}")
        elif self.split == 'val':
            print(f"Validation data shape: {self.val.shape}, mean: {np.mean(self.val, axis=0)}, std: {np.std(self.val, axis=0)}")
        elif self.split == 'test':
            print(f"Test data shape: {self.test.shape}, mean: {np.mean(self.test, axis=0)}, std: {np.std(self.test, axis=0)}")

    def __len__(self):
        if self.split == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.split == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.split == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.split == "train":
            inputs = self.train
            labels = self.train_labels
        elif self.split == 'val':
            inputs = self.val
            labels = self.val_labels
        elif self.split == 'test':
            inputs = self.test
            labels = self.test_labels
        
        return np.float32(inputs[index:index + self.win_size]), int(np.any(labels[index:index + self.win_size] == 1))


class PSMSegLoader(TSDataset):
    def __init__(self, cfg, split):
        super(PSMSegLoader, self).__init__(cfg, split)

    def _load_data(self):
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        test_label_path = os.path.join(self.data_dir, 'test_label.csv')

        train = pd.read_csv(train_path)
        train = train.values[:, 1:]
        train = np.nan_to_num(train)
        # placeholder for train_labels
        train_labels = np.zeros(train.shape[0])

        test = pd.read_csv(test_path)
        test = test.values[:, 1:]
        test = np.nan_to_num(test)

        # 尝试加载标签文件，如果不存在则默认为全 0
        if os.path.exists(test_label_path):
            test_labels = pd.read_csv(test_label_path).values[:, 1:]
        else:
            print(f"test_label.csv not found. Using all zeros as test labels.")
            test_labels = np.zeros(test.shape[0])

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class MSLSegLoader(TSDataset):
    def __init__(self, cfg, split):
        self.entity_lists = [
            "M-6", "M-1", "M-2", "S-2", "P-10", "T-4", "T-5", "F-7",
            "M-3", "M-4", "M-5", "P-15", "C-1", "C-2", "T-12", "T-13",
            "F-4", "F-5", "D-14", "T-9", "P-14", "T-8", "P-11", "D-15",
            "D-16", "M-7", "F-8"
        ]
        self.entity = cfg.DATA.NAME.split('_')[1]
        assert self.entity in self.entity_lists
        super(MSLSegLoader, self).__init__(cfg, split)

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'SMAP_MSL')
        
        train = np.load(os.path.join(self.data_dir, "train", f"{self.entity}.npy"))
        test = np.load(os.path.join(self.data_dir, "test", f"{self.entity}.npy"))
        
        label_file = os.path.join(self.data_dir, 'test', 'labeled_anomalies.csv')
        df = pd.read_csv(label_file)
        anomaly_sequences = eval(df.loc[df['chan_id'] == self.entity]['anomaly_sequences'].item())
        
        test_labels = np.zeros(test.shape[0])
        for start, end in anomaly_sequences:
            test_labels[start: end + 1] = 1
        test_labels = test_labels.astype(int)
        
        train_labels = np.zeros(train.shape[0])
        
        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class SMDSegLoader(TSDataset):
    def __init__(self, cfg, split):
        # 支持两种格式：SMD_1 或 SMD_1-1
        name_parts = cfg.DATA.NAME.split('_')[1:]
        if len(name_parts) == 1:
            # 单个数字，如 SMD_1，使用第一个文件
            self.entity = name_parts[0]
            self.file_suffix = f"1-{self.entity}"  # machine-1-1, machine-1-2, etc.
        else:
            # 完整格式，如 SMD_1-1
            self.entity = '-'.join(name_parts)
            self.file_suffix = self.entity
        super(SMDSegLoader, self).__init__(cfg, split)

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'ServerMachineDataset')

        # 尝试多种可能的文件名格式
        train_file = os.path.join(self.data_dir, 'preprocessed', f'machine-{self.file_suffix}_train.pkl')
        test_file = os.path.join(self.data_dir, 'preprocessed', f'machine-{self.file_suffix}_test.pkl')
        test_label_file = os.path.join(self.data_dir, 'preprocessed', f'machine-{self.file_suffix}_test_label.pkl')

        # 如果文件不存在，尝试查找第一个可用的文件
        if not os.path.exists(train_file):
            import glob
            train_files = glob.glob(os.path.join(self.data_dir, 'preprocessed', 'machine-*_train.pkl'))
            if train_files:
                train_file = train_files[0]
                # 提取对应的 test 和 label 文件
                base_name = train_file.replace('_train.pkl', '')
                test_file = base_name + '_test.pkl'
                test_label_file = base_name + '_test_label.pkl'
                print(f"Using first available SMD file: {train_file}")

        with open(train_file, 'rb') as f:
            train = pickle.load(f)
        with open(test_file, 'rb') as f:
            test = pickle.load(f)
        with open(test_label_file, 'rb') as f:
            test_labels = pickle.load(f)
        test_labels = test_labels.astype(int)
        # placeholder for train, val labels
        train_labels = np.zeros(train.shape[0])

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class SWaTSegLoader(TSDataset):
    def __init__(self, cfg, split):
        super(SWaTSegLoader, self).__init__(cfg, split)
    
    def _load_data(self):
        if self.downsample_rate > 1:
            try:
                return self._load_downsampled_data()
            except:
                print(f"downsampled data not found. Load original data instead and downsample.")

        # 支持两种格式：新格式 (SWaT_train.csv/SWaT_test.csv) 或旧格式
        train_csv_path = self.data_dir + "/SWaT_train.csv"
        train_csv_alt_path = self.data_dir + "/SWaT_Dataset_Normal_v1.csv"
        train_excel_path = self.data_dir + "/SWaT_Dataset_Normal_v1.xlsx"

        if os.path.exists(train_csv_path):
            train_df = pd.read_csv(train_csv_path)
            # 假设最后一列是标签列
            if 'label' in train_df.columns:
                train_labels = train_df['label'].to_numpy().astype(int)
                train_df = train_df.drop(columns=['label'])
            elif 'Normal/Attack' in train_df.columns:
                train_labels = (train_df['Normal/Attack'] == 'Attack').to_numpy().astype(int)
                train_df = train_df.drop(columns=['Normal/Attack'])
            else:
                train_labels = np.zeros(len(train_df))
            # 删除非特征列
            drop_cols = [c for c in ['Timestamp'] if c in train_df.columns]
            train_df = train_df.drop(columns=drop_cols)
            train_df = train_df.astype(np.float32)
        elif os.path.exists(train_csv_alt_path):
            train_df = pd.read_csv(train_csv_alt_path)
            train_df = train_df.iloc[1:, 1:-1]
            train_df = train_df.astype(np.float32)
            train_labels = np.zeros(len(train_df))
        elif os.path.exists(train_excel_path):
            train_df = pd.read_excel(train_excel_path)
            train_df.to_csv(train_csv_alt_path, index=False)
            train_df = train_df.iloc[1:, 1:-1]
            train_df = train_df.astype(np.float32)
            train_labels = np.zeros(len(train_df))
        else:
            raise FileNotFoundError(f"SWaT train file not found.")

        # Test data
        test_csv_path = self.data_dir + "/SWaT_test.csv"
        test_csv_alt_path = self.data_dir + "/SWaT_Dataset_Attack_v0.csv"
        test_excel_path = self.data_dir + "/SWaT_Dataset_Attack_v0.xlsx"

        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            # 优先使用 label 列，其次使用 Normal/Attack 列
            if 'label' in test_df.columns:
                test_labels = test_df['label'].to_numpy().astype(int)
            elif 'Normal/Attack' in test_df.columns:
                test_labels = (test_df['Normal/Attack'] == 'Attack').to_numpy().astype(int)
            else:
                test_labels = np.zeros(len(test_df))
            # 删除非特征列
            drop_cols = [c for c in ['label', 'Normal/Attack', 'Timestamp'] if c in test_df.columns]
            test_df = test_df.drop(columns=drop_cols)
            test_df = test_df.astype(np.float32)
        elif os.path.exists(test_csv_alt_path):
            test_df = pd.read_csv(test_csv_alt_path)
            test_labels = (test_df.iloc[:, -1] == 'Attack').to_numpy().astype(int)
            test_df = test_df.iloc[1:, 1:-1]
            test_df = test_df.astype(np.float32)
        elif os.path.exists(test_excel_path):
            test_df = pd.read_excel(test_excel_path)
            test_df.to_csv(test_csv_alt_path, index=False)
            test_labels = (test_df.iloc[:, -1] == 'Attack').to_numpy().astype(int)
            test_df = test_df.iloc[1:, 1:-1]
            test_df = test_df.astype(np.float32)
        else:
            raise FileNotFoundError(f"SWaT test file not found.")
        
        self.var_names = list(train_df.columns)
        
        train = train_df.values
        train_labels = np.zeros(train.shape[0])
        test = test_df.values
        
        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()
        
        if self.downsample_rate > 1:
            train, val, test = downsample(train, self.downsample_rate), downsample(val, self.downsample_rate), downsample(test, self.downsample_rate)
            train_labels, val_labels, test_labels = downsample(train_labels, self.downsample_rate), downsample(val_labels, self.downsample_rate), downsample(test_labels, self.downsample_rate)
            self._save_downsampled_data(train, val, test, train_labels, val_labels, test_labels)
        
        return train, val, test, train_labels, val_labels, test_labels


class WADISegLoader(TSDataset):
    def __init__(self, cfg, split):
        super(WADISegLoader, self).__init__(cfg, split)
    
    def _load_data(self):
        if self.downsample_rate > 1:
            try:
                return self._load_downsampled_data()
            except:
                print(f"downsampled data not found. Load original data instead and downsample.")

        # 支持新格式：WADI_train.csv / WADI_test.csv
        train_csv_path = self.data_dir + "/WADI_train.csv"
        test_csv_path = self.data_dir + "/WADI_test.csv"

        if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)

            # 训练数据：无标签列（全部正常）
            train_labels = np.zeros(len(train_df))

            # 测试数据：label 列 (0=Normal, 1=Attack)
            if 'label' in test_df.columns:
                test_labels = test_df['label'].to_numpy().astype(int)
                test_df = test_df.drop(columns=['label'])
            else:
                test_labels = np.zeros(len(test_df))

            # 删除非特征列 (Time, Timestamp 等)
            drop_cols = [c for c in ['Time', 'Timestamp', 'timestamp'] if c in test_df.columns]
            test_df = test_df.drop(columns=drop_cols)

            train_df = train_df.dropna(axis='columns', how='all').dropna().astype(np.float32)
            test_df = test_df.dropna(axis='columns', how='all').dropna().astype(np.float32)

        else:
            # 原始格式
            train_df = pd.read_csv(self.data_dir + "/WADI.A2_19 Nov 2019/WADI_14days_new.csv")
            train_df = train_df.dropna(axis='columns', how='all').dropna()
            train_df = train_df.iloc[:, 3:].astype(np.float32)
            test_df = pd.read_csv(self.data_dir + "/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv", header=1)
            test_df = test_df.dropna(axis='columns', how='all').dropna()
            test_df = test_df.iloc[:, 3:].astype(np.float32)

            self.var_names = list(train_df.columns)
            test_labels = test_df['Attack LABLE (1:No Attack, -1:Attack)'].values
            test_labels = (test_labels == -1.0).astype(int)
            test = test_df.drop(columns='Attack LABLE (1:No Attack, -1:Attack)').values
            train = train_df.values
            train_labels = np.zeros(train.shape[0])

            if self.train_ratio < 1.0:
                train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
            else:
                val, val_labels = test.copy(), test_labels.copy()

            return train, val, test, train_labels, val_labels, test_labels

        self.var_names = list(train_df.columns)
        train = train_df.values
        test = test_df.values

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()
        
        if self.downsample_rate > 1:
            train, val, test = downsample(train, self.downsample_rate), downsample(val, self.downsample_rate), downsample(test, self.downsample_rate)
            train_labels, val_labels, test_labels = downsample(train_labels, self.downsample_rate), downsample(val_labels, self.downsample_rate), downsample(test_labels, self.downsample_rate)
            self._save_downsampled_data(train, val, test, train_labels, val_labels, test_labels)
        
        return train, val, test, train_labels, val_labels, test_labels


class Lorenz96SegLoader(TSDataset):
    def __init__(self, cfg, split):
        # 解析 DATA.NAME: Lorenz96_point_global_outliers_factor4.0
        # 文件名格式：test_point_global_outliers_factor4.0.npy
        name_suffix = cfg.DATA.NAME[len('Lorenz96_'):]
        # 找到 '_factor' 位置
        idx = name_suffix.rfind('_factor')
        if idx >= 0:
            self.anomaly_type = name_suffix[:idx]  # 如 point_global_outliers
            factor_val = name_suffix[idx+len('_factor'):]  # 如 4.0
            if factor_val == 'None':
                self.factor_str = 'None'
            else:
                self.factor_str = factor_val
        else:
            self.anomaly_type = name_suffix
            self.factor_str = 'None'
        super(Lorenz96SegLoader, self).__init__(cfg, split)
        self.true_cm = np.load(os.path.join(self.data_dir, 'GC.npy'))

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'Lorenz96')
        train = np.load(os.path.join(self.data_dir, 'train.npy'))
        train_labels = np.zeros(train.shape[0])

        # 构建文件名：test_point_contextual_outliers_factor4.0.npy
        test_file = f'test_{self.anomaly_type}_factor{self.factor_str}.npy'
        test = np.load(os.path.join(self.data_dir, test_file))

        test_labels_file = f'test_{self.anomaly_type}_factor{self.factor_str}_labels.npy'
        test_labels = np.load(os.path.join(self.data_dir, test_labels_file))

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


class VARSegLoader(TSDataset):
    def __init__(self, cfg, split):
        # 解析 DATA.NAME: VAR_point_global_outliers_factor4.0
        # 文件名格式：test_point_global_outliers_factor4.0.npy
        name_suffix = cfg.DATA.NAME[len('VAR_'):]
        idx = name_suffix.rfind('_factor')
        if idx >= 0:
            self.anomaly_type = name_suffix[:idx]  # 如 point_global_outliers
            factor_val = name_suffix[idx+len('_factor'):]  # 如 4.0
            if factor_val == 'None':
                self.factor_str = 'None'
            else:
                self.factor_str = factor_val
        else:
            self.anomaly_type = name_suffix
            self.factor_str = 'None'
        super(VARSegLoader, self).__init__(cfg, split)
        self.true_cm = np.load(os.path.join(self.data_dir, 'GC.npy'))

    def _load_data(self):
        self.data_dir = os.path.join(self.cfg.DATA.BASE_DIR, 'VAR')
        train = np.load(os.path.join(self.data_dir, 'train.npy'))
        train_labels = np.zeros(train.shape[0])

        # 构建文件名：test_point_global_outliers_factor4.0.npy
        test_file = f'test_{self.anomaly_type}_factor{self.factor_str}.npy'
        test = np.load(os.path.join(self.data_dir, test_file))

        test_labels_file = f'test_{self.anomaly_type}_factor{self.factor_str}_labels.npy'
        test_labels = np.load(os.path.join(self.data_dir, test_labels_file))

        if self.train_ratio < 1.0:
            train, val, train_labels, val_labels = self._split_train_val(train, train_labels)
        else:
            val, val_labels = test.copy(), test_labels.copy()

        return train, val, test, train_labels, val_labels, test_labels


def build_dataset(cfg, split):
    dataset_name = cfg.DATA.NAME

    dataset_loaders = {
        "SMD": SMDSegLoader,
        "MSL": MSLSegLoader,
        "PSM": PSMSegLoader,
        "SWaT": SWaTSegLoader,
        "WADI": WADISegLoader,
        "Lorenz96": Lorenz96SegLoader,
        "VAR": VARSegLoader,
    }

    for key in dataset_loaders:
        if key in dataset_name:
            return dataset_loaders[key](cfg, split)

    raise ValueError(f"Unknown dataset name: {dataset_name}")
