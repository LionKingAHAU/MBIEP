import numpy as np


class getData(object):
    def __init__(self):
        # 1. import the dataset with trainset : validset : testset = 0.6 : 0.2 : 0.2
        self.train_emb = np.load('dataset/trainset/train_emb.npy')
        self.train_sub = np.load('dataset/trainset/train_sub.npy')
        self.train_gse = np.load('dataset/trainset/train_gse.npy')
        self.train_label = np.load('dataset/trainset/train_label.npy')

        self.test_E = np.load('dataset/testset/test_emb.npy')
        self.test_S = np.load('dataset/testset/test_sub.npy')
        self.test_G = np.load('dataset/testset/test_gse.npy')
        self.test_Y = np.load('dataset/testset/test_label.npy')

        self.vali_emb = np.load('dataset/valiset/vali_emb.npy')
        self.vali_sub = np.load('dataset/valiset/vali_sub.npy')
        self.vali_gse = np.load('dataset/valiset/vali_gse.npy')
        self.vali_label = np.load('dataset/valiset/vali_label.npy')

        print("training data numbers(%d%%): %d" % (60, len(self.train_label)))
        # 2. strip the pos and neg index
        self.pos_idx = (self.train_label == 1).reshape(-1)
        self.neg_idx = (self.train_label == 0).reshape(-1)

        # 3. get the size of train set and print the information
        self.training_size = len(self.train_label[self.pos_idx]) * 2
        print("positive data numbers", str(self.training_size // 2))
        print("negative data numbers", str(len(self.neg_idx)))

    def shuffle(self):
        # 1. shuffle the negative part
        mark = list(range(int(np.sum(self.neg_idx))))
        np.random.shuffle(mark)

        # 2. even the neg and pos num in the train set
        self.train_E = np.concatenate(
            [self.train_emb[self.pos_idx], self.train_emb[self.neg_idx][mark][:self.training_size // 2]])
        self.train_G = np.concatenate(
            [self.train_gse[self.pos_idx], self.train_gse[self.neg_idx][mark][:self.training_size // 2]])
        self.train_S = np.concatenate(
            [self.train_sub[self.pos_idx], self.train_sub[self.neg_idx][mark][:self.training_size // 2]])
        self.train_Y = np.concatenate(
            [self.train_label[self.pos_idx], self.train_label[self.neg_idx][mark][:self.training_size // 2]])

        # 3. shuffle the train set concatenated above
        mark = list(range(self.training_size))
        np.random.shuffle(mark)
        self.train_E = self.train_E[mark]
        self.train_G = self.train_G[mark]
        self.train_S = self.train_S[mark]
        self.train_Y = self.train_Y[mark]

    def vali_shuffle(self):
        # 4.shuffle the validate set
        mark = list(range(len(self.vali_label)))
        np.random.shuffle(mark)
        self.vali_E = self.vali_emb[mark]
        self.vali_G = self.vali_gse[mark]
        self.vali_S = self.vali_sub[mark]
        self.vali_Y = self.vali_label[mark]


if __name__ == '__main__':
    data = getData()
