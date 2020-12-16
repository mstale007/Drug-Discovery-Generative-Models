from tensorflow.keras.utils import Sequence
from smile_tokenizer import SmilesTokenizer
import numpy as np
from tqdm import tqdm

class Sequence_DataLoader(Sequence):
    def __init__(self, data_filename,validation_split=0.1, seed=0, batch_size=1,data_type='train'):
        self.data_filename=data_filename
        self.data_type = data_type
        assert self.data_type in ['train', 'valid', 'finetune']

        self.max_len = 0

        if self.data_type == 'train':
            self.smiles = self._load(self.data_filename)
        elif self.data_type == 'finetune':
            self.smiles = self._load(self.finetune_data_filename)
        else:
            pass

        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict

        self.tokenized_smiles = self._tokenize(self.smiles)
        self.seed=seed
        self.batch_size=batch_size
        self.train_smi_max_len=47

        self.validation_split=validation_split
        if self.data_type in ['train', 'valid']:
            self.idx = np.arange(len(self.tokenized_smiles))
            #print(self.idx)
            self.valid_size = int(
                np.ceil(
                    len(self.tokenized_smiles) * self.validation_split))
            np.random.seed(self.seed)
            np.random.shuffle(self.idx)
            

    def _set_data(self):
        if self.data_type == 'train':
            ret = [
                self.tokenized_smiles[self.idx[i]]
                for i in self.idx[self.valid_size:]
            ]
        elif self.data_type == 'valid':
            ret = [
                self.tokenized_smiles[self.idx[i]]
                for i in self.idx[:self.valid_size]
            ]
        else:
            ret = self.tokenized_smiles
        return ret

    def _load(self, data_filename):
        print('Loading Smiles...:)')
        with open(data_filename) as f:
            smiles = [s.rstrip() for s in f]
        print('Done.')
        return smiles[:7000]

    def _tokenize(self, smiles):
        assert isinstance(smiles, list)
        print('Tokenizing Smiles...:)')
        tokenized_smiles = [self.st.tokenize(smi) for smi in tqdm(smiles)]

        if self.data_type == 'train':
            for tokenized_smi in tokenized_smiles:
                length = len(tokenized_smi)
                if self.max_len < length:
                    self.max_len = length
            self.train_smi_max_len = self.max_len
        print('Done.')
        return tokenized_smiles

    def __len__(self):
        target_tokenized_smiles = self._set_data()
        if self.data_type in ['train', 'valid']:
            ret = int(
                np.ceil(
                    len(target_tokenized_smiles) /
                    float(self.batch_size)))
        else:
            ret = int(
                np.ceil(
                    len(target_tokenized_smiles) /
                    float(self.finetune_batch_size)))
        return ret

    def __getitem__(self, idx):
        target_tokenized_smiles = self._set_data()
        if self.data_type in ['train', 'valid']:
            data = target_tokenized_smiles[idx *
                                           self.batch_size:(idx + 1) *
                                           self.batch_size]
        else:
            data = target_tokenized_smiles[idx *
                                           self.finetune_batch_size:
                                           (idx + 1) *
                                           self.finetune_batch_size]
        data = self._padding(data)

        self.X, self.y = [], []
        for tp_smi in data:
            X = [self.one_hot_dict[symbol] for symbol in tp_smi[:-1]]
            self.X.append(X)
            y = [self.one_hot_dict[symbol] for symbol in tp_smi[1:]]
            self.y.append(y)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        #print("X: ",self.X)
        #print("Y: ",self.y)
        return self.X, self.y

    def _pad(self, tokenized_smi):
        return ['G'] + tokenized_smi + ['E'] + [
            'A' for _ in range(self.max_len - len(tokenized_smi))
        ]

    def _padding(self, data):
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        return padded_smiles
