import torch.utils.data
import pandas as pd
import torch


class CastomDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, cnt_attributes: int, batch_size: int,
                 indices: list = None, num_samples: int = None, balanced: bool = False):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        self.batch_size = batch_size
        self.cnt_attributes = cnt_attributes
        self.labels = self._get_labels(dataset)
        self.attribute_names = self.labels.columns
        print(self.labels)

        dfs = []
        weights_all = []
        for attribute_id in range(cnt_attributes):
            df = pd.DataFrame()
            current_attribute = self.attribute_names[attribute_id]
            df["label"] = self.labels[current_attribute]
            df = df[df["label"] != -1]
            # Compute sample weights
            label_to_count = df["label"].value_counts()
            weights = 1.0 / label_to_count[df["label"]]
            weights = torch.DoubleTensor(weights.to_list())
            weights_all.append(weights)
            dfs.append(df)

        self.dfs = dfs
        self.weights = weights_all
        self.balanced = balanced

    def _get_labels(self, dataset):
        if isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        samples_cnt = 0
        attribute_id = 0
        while samples_cnt < len(self.indices):
            attribute_id = (attribute_id + 1) % self.cnt_attributes
            current_attribute = self.attribute_names[attribute_id]
            df = self.dfs[current_attribute]
            # Generate balanced batch
            if self.balanced:
                generated = torch.multinomial(self.weights[current_attribute], self.batch_size, replacement=True)
            else:
                generated = torch.multinomial(torch.ones(len(self.weights[current_attribute])), self.batch_size,
                                              replacement=False)

            for j in range(self.batch_size):
                pos = generated[j]
                yield df.index[pos]
                samples_cnt += 1
                if samples_cnt == len(self.indices):
                    break

    def __len__(self):
        return self.num_samples
