# PCA based coding
import numpy as np
from sklearn.decomposition import PCA





class PCA():
    def __init__(self):
        n_components = 100
        compressor = PCA(n_components)

    def learn(self, raw_datas):
        compressor.fit(raw_datas)


    def compress(self, raw_data):
        return compressor.fit_transform(raw_data)

    def decompress(self, data):
        return compressor.inverse_transform(data)

    def compress_multiple(self):
        pass

    def decompress_multiple(self):
        pass

