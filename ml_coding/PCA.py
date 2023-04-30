# PCA based coding
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class PCA_compression():
    def __init__(self):
        self.n_components = 30
        self.compressor = PCA(self.n_components)

    def learn(self, raw_datas):
        return self.compressor.fit(raw_datas)
    
    def scree(self, raw_data, var_per=94, scree_plot=False):
        pca = PCA()
        pca.fit(raw_data)
        
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)*100
        k = np.argmax(cumulative_var>var_per)
        
        if scree_plot:
            plt.figure(figsize=[10,5])
            plt.title('Cumulative Explained Variance explained by the components')
            plt.ylabel('Cumulative Explained variance')
            plt.xlabel('Principal components')
            plt.axvline(x=k, color="k", linestyle="--")
            plt.axhline(y=var_per, color="r", linestyle="--")
            ax = plt.plot(cumulative_var)
        
        return k
        
    def compress(self, raw_data):
        return self.compressor.fit_transform(raw_data)

    def decompress(self, data):
        return self.compressor.inverse_transform(data)

    def compress_multiple(self, images):
        comp_images = []
        comp_images = [self.compress(images[i]) for i in range(len(images))]
        return np.array(comp_images)

    def decompress_multiple(self, comp_images):
        recon_images = []
        recon_images = [self.decompress(comp_images[i]) for i in range(len(comp_images))]
        return np.array(recon_images)
