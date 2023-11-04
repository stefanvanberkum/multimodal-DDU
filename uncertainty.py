"""Uncertainty estimation methods."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy, multivariate_normal
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture


class DDU:
    def __init__(self, z, y):
        """Deep deterministic uncertainty predictor.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param y: A label vector of size n_class.
        """

        self.classes, counts = np.unique(y, return_counts=True)
        self.log_prior = dict()
        self.components = dict()
        self.n_class = len(self.classes)
        for i in range(self.n_class):
            c = self.classes[i]

            # Compute prior.
            self.log_prior[c] = np.log(counts[i] / len(y))

            # Compute GMM component.
            subset = z[y == c, :]
            mu = np.mean(subset, axis=0)
            Sigma = np.cov(subset, rowvar=False)
            self.components[c] = multivariate_normal(mu, Sigma)

    def predict(self, z, p):
        """Predict uncertainty for one sample.

        Uses softmax entropy for aleatoric uncertainty and a GMM for epistemic uncertainty.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param p: An output prediction matrix of shape (n_obs, n_class).
        :return: A tuple with uncertainties (aleatoric, epistemic).
        """

        aleatoric, _ = softmax_uncertainty(p)

        n_obs = len(z)
        log_probs = np.zeros((n_obs, self.n_class))
        for i in range(n_obs):
            sample = z[i, :]

            for j in range(self.n_class):
                c = self.classes[j]
                log_probs[i, j] = self.log_prior[c] + self.components[c].logpdf(sample)
        epistemic = np.exp(logsumexp(log_probs, axis=1))

        return aleatoric, epistemic


class DDU_KD:
    def __init__(self, z):
        """Deep deterministic uncertainty predictor with kernel density estimation.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        """

        self.kde = KernelDensity().fit(z)

    def predict(self, z, p):
        """Predict uncertainty for one sample.

        Uses softmax entropy for aleatoric uncertainty and a kernel density estimator for epistemic uncertainty.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param p: An output prediction matrix of shape (n_obs, n_class).
        :return: A tuple with uncertainties (aleatoric, epistemic).
        """

        aleatoric, _ = softmax_uncertainty(p)
        epistemic = np.exp(self.kde.score_samples(z))

        return aleatoric, epistemic


class DDU_CWKD:
    def __init__(self, z, y):
        """Deep deterministic uncertainty predictor with component-wise kernel density estimation.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param y: A label vector of size n_class.
        """

        self.classes, counts = np.unique(y, return_counts=True)
        self.log_prior = dict()
        self.components = dict()
        self.n_class = len(self.classes)
        for i in range(self.n_class):
            c = self.classes[i]

            # Compute prior.
            self.log_prior[c] = np.log(counts[i] / len(y))

            # Compute mixture component.
            subset = z[y == c, :]
            self.components[c] = KernelDensity.fit(subset)

    def predict(self, z, p):
        """Predict uncertainty for one sample.

        Uses softmax entropy for aleatoric uncertainty and a mixture model for epistemic uncertainty.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param p: An output prediction matrix of shape (n_obs, n_class).
        :return: A tuple with uncertainties (aleatoric, epistemic).
        """

        aleatoric, _ = softmax_uncertainty(p)

        n_obs = len(z)
        log_probs = np.zeros((n_obs, self.n_class))
        for i in range(n_obs):
            sample = z[i, :]

            for j in range(self.n_class):
                c = self.classes[j]
                log_probs[i, j] = self.log_prior[c] + self.components[c].score_samples(sample)
        epistemic = np.exp(logsumexp(log_probs, axis=1))

        return aleatoric, epistemic


class DDU_VI:
    def __init__(self, z, n_components):
        """Deep deterministic uncertainty predictor with a Dirichlet process mixture.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param n_components: The maximum number of components.
        """

        self.gmm = BayesianGaussianMixture(n_components=n_components).fit(z)

    def predict(self, z, p):
        """Predict uncertainty for one sample.

        Uses softmax entropy for aleatoric uncertainty and a kernel density estimator for epistemic uncertainty.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param p: An output prediction matrix of shape (n_obs, n_class).
        :return: A tuple with uncertainties (aleatoric, epistemic).
        """

        aleatoric, _ = softmax_uncertainty(p)
        epistemic = np.exp(self.gmm.score_samples(z))

        return aleatoric, epistemic


def softmax_uncertainty(p):
    """Softmax entropy for both aleatoric and epistemic uncertainty.

    :param p: An output prediction matrix of shape (n_obs, n_class).
    :return: A tuple with uncertainties (aleatoric, epistemic).
    """

    aleatoric = entropy(p, axis=1)
    epistemic = aleatoric
    return aleatoric, epistemic



