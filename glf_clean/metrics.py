from scipy import linalg
import numpy as np


def frechet_distance(latent_true: np.ndarray, latent_artificial: np.ndarray) -> float:
    assert latent_true.shape == latent_artificial.shape

    mu_true, sigma_true = np.mean(latent_true, axis=0), np.cov(latent_true, rowvar=False)
    mu_artificial, sigma_artificial = np.mean(latent_artificial, axis=0), np.cov(latent_artificial, rowvar=False)

    matrix_sqrt = linalg.sqrtm(np.dot(sigma_artificial, sigma_true))
    assert np.isfinite(matrix_sqrt).all(), 'https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py'
    assert not np.iscomplexobj(matrix_sqrt), 'https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py'

    euc_norm_squared = np.linalg.norm(mu_true - mu_artificial, ord=2) ** 2
    fid = euc_norm_squared + np.trace(sigma_true) + np.trace(sigma_artificial) - 2 * np.trace(matrix_sqrt)
    return float(fid)


if __name__ == '__main__':

    latent_true = np.random.rand(1024, 32)
    latent_art = np.random.rand(1024, 32)

    pos_fid = frechet_distance(latent_true, latent_art)
    zero_fid = frechet_distance(latent_true, latent_true)
    print(f'Positive FID = {pos_fid}')
    print(f'Zero FID = {zero_fid}')
