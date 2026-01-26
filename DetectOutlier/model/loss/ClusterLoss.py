import torch
import torch.nn as nn
import math
import time
import numpy as np
import faiss

class ClusterLoss(nn.Module):
    def __init__(self, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = 2
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return {
            "loss_v": loss + ne_loss
        }


class KmeansFeature:
    #@staticmethod
    def __init__(self, k=2, norm=False):
        self.k = k
        self.norm = norm

    def preprocess_features(self, npdata, norm=False):
        """Preprocess an array of features.
        Args:
            npdata (np.array N * ndim): features to preprocess
            pca (int): dim of output
        Returns:
            np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
        """
        _, ndim = npdata.shape
        npdata = npdata.astype('float32')
        '''
        # Apply PCA-whitening with Faiss
        mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
        '''
        # L2 normalization
        if norm:
            row_sums = np.linalg.norm(npdata, axis=1)
            npdata = npdata / row_sums[:, np.newaxis]

        return npdata

    def run_kmeans(self, x, device, min_num=100, max_num=20000, verbose=False):
        """Runs kmeans on 1 GPU.
        Args:
            x: data
            k (int): number of clusters
        Returns:
            list: ids of data in each cluster
        """
        n_data, d = x.shape

        # faiss implementation of k-means
        clus = faiss.Clustering(d, self.k)
        clus.niter = 20
        clus.max_points_per_centroid = max_num
        clus.min_points_per_centroid = min_num
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = device
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        clus.train(x, index)
        _, I = index.search(x, 1)
        stats = clus.iteration_stats
        losses = np.array([
            stats.at(i).obj for i in range(stats.size())
        ])
        # losses = faiss.vector_to_array(clus.obj)
        if verbose:
            print('k-means loss evolution: {0}'.format(losses))

        return [int(n[0]) for n in I], losses[-1]

    #@staticmethod
    def cluster(self, data, device=0, min_num=100, max_num=20000, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = self.preprocess_features(data, norm=self.norm)

        # cluster the data
        I, loss = self.run_kmeans(xb, device, min_num, max_num, verbose)
        self.featuers_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.featuers_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss