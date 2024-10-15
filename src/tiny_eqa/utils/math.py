from sklearn.decomposition import PCA

from tiny_eqa.data.common import NumpyTensor, TorchTensor


def min_max_norm(tensor: NumpyTensor | TorchTensor, dim=None):
    """
    """
    tmin = tensor.min(axis=dim, keepdims=True)
    tmax = tensor.max(axis=dim, keepdims=True)
    return (tensor - tmin) / (tmax - tmin)


def pca_transform(tensor: NumpyTensor['batch', 'dim'], n: int) -> NumpyTensor['batch', 'n']:
    """
    """
    pca = PCA(n_components=n)
    pca.fit(tensor)
    return pca.transform(tensor)