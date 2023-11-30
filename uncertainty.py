"""Uncertainty estimation methods.

Each of the epistemic uncertainty estimation methods returns the negative log density (except softmax, this returns
entropy)."""

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

            jitters = [0,  np.finfo(np.double).smallest_normal] + [10 ** exp for exp in range(-308, 0, 1)]
            for jitter in jitters:
                try:
                    self.components[c] = multivariate_normal(mu, Sigma + jitter * np.eye(Sigma.shape[0]))
                except np.linalg.LinAlgError as e:
                    if "symmetric positive definite" in str(e):
                        continue
                break

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
        epistemic = -logsumexp(log_probs, axis=1)

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
        epistemic = -self.kde.score_samples(z)

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
            self.components[c] = KernelDensity().fit(subset)

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
            sample = np.expand_dims(z[i, :], axis=0)

            for j in range(self.n_class):
                c = self.classes[j]
                log_probs[i, j] = self.log_prior[c] + self.components[c].score_samples(sample)
        epistemic = -logsumexp(log_probs, axis=1)

        return aleatoric, epistemic


class DDU_VI:
    def __init__(self, z, n_components, scaling_parameter=None):
        """Deep deterministic uncertainty predictor with a Dirichlet process mixture.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param n_components: The maximum number of components.
        :param scaling_parameter: The weight concentration prior.
        """

        self.gmm = BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=scaling_parameter).fit(z)

    def predict(self, z, p):
        """Predict uncertainty for one sample.

        Uses softmax entropy for aleatoric uncertainty and a kernel density estimator for epistemic uncertainty.

        :param z: A feature matrix z of shape (n_obs, n_feat).
        :param p: An output prediction matrix of shape (n_obs, n_class).
        :return: A tuple with uncertainties (aleatoric, epistemic).
        """

        aleatoric, _ = softmax_uncertainty(p)
        epistemic = -self.gmm.score_samples(z)

        return aleatoric, epistemic


def softmax_uncertainty(p):
    """Softmax entropy for both aleatoric and epistemic uncertainty.

    :param p: An output prediction matrix of shape (n_obs, n_class).
    :return: A tuple with uncertainties (aleatoric, epistemic).
    """

    aleatoric = entropy(p, axis=1)
    epistemic = aleatoric
    
    return aleatoric, epistemic
