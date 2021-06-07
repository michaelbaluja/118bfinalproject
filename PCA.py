import numpy as np
from sklearn.preprocessing import StandardScaler

class PCA():
    def __init__(self, n_components, scale=False):
        '''
        Creates instance of PCA tranformer. Class structure loosely follows scikit-learn schema. 
        Implementation follows formulation from Bishop Pattern Recognition

        Params:
        - n_components (int): number of principal components to keep.
        - scale (bool) (default=False): flag for whether or not to scale the data before fitting/transforming
        '''
        self.n_components = n_components
        self.scale = scale

        if self.scale:
            self.scaler = StandardScaler()

    def power_eig(self, A, n_iterations=100):
        '''
        Utilizes the power method to compute the dominant eigenvector of a matrix. Continues
        iteration for a set number of cycles

        Params:
        - A (np.array): the matrix to calculate the dominant eigenvector of
        - n_iterations (int) (default=100): threshold for iteration continuation

        Returns:
        - q (np.array): the dominant eigenvector
        - eig_val (float): the dominant eigenvalue
        '''

        q = np.random.rand(A.shape[1])
        for _ in range(n_iterations):
            q = np.dot(A, q)
            eig_val = np.linalg.norm(q, ord=np.inf)
            q /= eig_val

        return q, eig_val

    def compute_mean(self, X, axis=0):
        '''
        Calculates and returns the mean of the data for given data points
        
        Params:
        - X (np.array): data points to compute mean of
        - axis (int) (default=0 (rows)): axis to take mean over 

        Returns:
        - x_bar (np.array): mean data point vector
        '''
        # Make sure that a valid axis was passed in
        assert axis in range(len(X.shape))

        x_bar = np.mean(X, axis=axis)
        return x_bar


    def compute_covariance_matrix(self, X):
        '''
        Creates and returns the covariance matrix for given data points

        Params:
        - X (np.array): array of data points

        Returns:
        - S (np.array): covariance matrix of data points
        '''
        N, M = X.shape
        S = np.zeros([M,M])

        if self.scale:
            x_bar = 0
        else:
            x_bar = self.compute_mean(X, axis=0)

        for n in range(N):
            S = S + np.matmul((X[n] - x_bar).reshape(M, 1), (X[n] - x_bar).reshape(M, 1).T)
        
        S /= N

        return S

    def orthonormalize_vectors(self):
        '''
        Performs modified Gram Schmidt process on the principal components so they are orthonormal.
        Pseudocode has been copied from Watkins' Fundamentals of Matrix Computation
        '''
        N = len(self.principal_components[0])
        r = np.zeros([N,N])
        v = self.principal_components
        for k in range(self.principal_components):
            for i in range(k):
                r[i, k] = np.dot(v[i], v[k])
                v[k] -= v[i] * r[i, k]            

            r[k, k] = np.linalg.norm(v[k])
            assert r[k, k] != 0, "Vectors are dependent"
            v[k] /= r[k, k]
        
        self.principal_components = v

    def sort_vectors(self):
        '''
        Sorts the principal_component (eigenvector) and eigenvalue lists based on eigenvalue magnitude
        '''
        # Sort the eigenvalues from largest to smallest. Store the sorted
        # eigenvalues in the column vector lambd.
        lohival = np.sort(self.eigenvalues)
        lohiindex = np.argsort(self.eigenvalues)
        lambd = np.flip(lohival)
        index = np.flip(lohiindex)
        self.eigenvalues = np.diag(lambd)
        
        # Sort eigenvectors to correspond to the ordered eigenvalues. Store sorted
        # eigenvectors as columns of the matrix vsort.
        M = self.principal_components[0].size[0]
        N = self.n_components
        Vsort = np.zeros((M, N))
        for i in range(N):
            Vsort[:,i] = self.principal_components[:,index[i]]
        
        self.principal_components = Vsort

    def fit(self, X):
        '''
        Given data points, fits the PCA model 

        Params:
        - X (np.array): data points to fit model to
        '''
        self.principal_components = []
        self.eigenvalues = []
        
        # Zero mean the data (if necessary)
        if self.scale:
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        # Compute the covariance matrix
        S = self.compute_covariance_matrix(X)

        # Find covariance eigenvectors
        for _ in range(self.n_components):
            eig_vect, eig_val = self.power_eig(S)
            self.principal_components.append(eig_vect.reshape(-1,1))
            self.eigenvalues.append(eig_val)

            # Creates vector x such that dot(x, eig_val) = 1
            x = np.ones([eig_vect.shape[0]])
            x /= np.dot(x, eig_vect)

            assert np.isclose(np.dot(x, eig_vect), 1), "Computation Error: eigenvector normalization is bust"
            assert x.shape == eig_vect.shape, "Sizing error with eigenvector"

            # Remove dominant eigenvector, allowing next iteration to find next largest eigenvector
            S = S - (eig_val * np.matmul(eig_vect, x.T))

        # Orthonormalizes the previously found eigenvectors
        self.orthonormalize_vectors

        # Rearrange eigenvectors in order of decreasing eigenvalue
        self.sort_vectors
    
    def transform(self, X):
        '''
        Transforms the inputted data based on the data that the model was fit to.

        Params:
        - X (np.array): array of data points to transform

        Returns:
        - X_transform (np.array): transformed data
        '''
        assert self.principal_components, "Error: Model has not been fit to data yet."

        # Subtract off mean of data
        if self.scale:
            X_transform = self.scaler.transform(X)

        # Create projection matrix
        P = np.hstack(self.principal_components)

        # Transform data
        X_transform = np.dot(X, P)

        return X_transform

    def fit_transform(self, X):
        '''
        Fits PCA model our data X, and then transforms the data. 
        pca.fit_transform(X) is equivalent to pca.fit(X) followed by pca.transform(X)
        
        Params:
        - X (np.array): data to fit model to and then transform

        Returns:
        - X_transform (np.array): transformed data
        '''
        self.fit(X)
        X_transform = self.transform(X)
        return X_transform