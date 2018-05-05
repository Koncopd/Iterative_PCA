import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

class likeAnnData:
    def setX(self, X):
        self.X = X

    def Chunked(self, chunks):
        start = 0
        for i in range(chunks):
            stop = start + len(self.X[i::chunks])
            yield self.X[start:stop]
            start = stop

D = likeAnnData()

D.setX(np.random.rand(100000, 1000))

n_comp = 80
n_chunks = 100

batch_size = 2000

ipca = IncrementalPCA(n_components=n_comp, batch_size = batch_size)

print('Training IPCA')

for chunk in D.Chunked(n_chunks):
    ipca.partial_fit(chunk)

OutIPCA = np.array([])

print('Fitting IPCA')

for chunk in D.Chunked(n_chunks):
    Tr = ipca.transform(chunk)
    OutIPCA = np.vstack([OutIPCA, Tr]) if OutIPCA.size else Tr

print('Training and fitting PCA')

pca = PCA(n_components=n_comp)
OutPCA = pca.fit_transform(D.X)

print('Squared Errors:')

print('Squared Error for PCA ', np.linalg.norm(D.X - pca.inverse_transform(OutPCA)))
print('Squared Error for IPCA ', np.linalg.norm(D.X - ipca.inverse_transform(OutIPCA)))
