from scipy import linalg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glf.metrics.inception import InceptionV3


# Maps feature dimensionality to their output blocks indices
DIM_TO_BLOCK_ID = {
    64: 0,  # First max pooling features
    192: 1,  # Second max pooling featurs
    768: 2,  # Pre-aux classifier features
    2048: 3  # Final average pooling features
}


class InceptionPredictor(nn.Module):
    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        assert output_dim in DIM_TO_BLOCK_ID
        self.output_dim = output_dim
        self.block_id = DIM_TO_BLOCK_ID[self.output_dim]
        self.model = InceptionV3([self.block_id])

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        out = self.model(imgs)[0]

        if out.shape[2] != 1 or out.shape[3] != 1:
            out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
        out = out.reshape(imgs.size(0), -1)
        return out


# https://arxiv.org/pdf/1706.08500.pdf (FID)
# https://arxiv.org/pdf/1606.03498.pdf (IS)
def frechet_distance(latent_true: np.ndarray, latent_artificial: np.ndarray) -> float:
    # https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
    assert latent_true.shape == latent_artificial.shape

    mu_true, sigma_true = np.mean(latent_true, axis=0), np.cov(latent_true, rowvar=False)
    mu_artificial, sigma_artificial = np.mean(latent_artificial, axis=0), np.cov(latent_artificial, rowvar=False)

    matrix_sqrt = linalg.sqrtm(np.dot(sigma_artificial, sigma_true))
    # numerical issues
    if not np.isfinite(matrix_sqrt).all():
        offset = np.eye(sigma_true.shape[0]) * 1e-8
        matrix_sqrt = linalg.sqrtm((sigma_true + offset).dot(sigma_artificial + offset))

    # numerical issues
    if np.iscomplexobj(matrix_sqrt):
        if not np.allclose(np.diagonal(matrix_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(matrix_sqrt.imag))
            raise ValueError('Imaginary component {}'.format(m))
        matrix_sqrt = matrix_sqrt.real

    euc_norm_squared = np.linalg.norm(mu_true - mu_artificial, ord=2) ** 2
    fid = euc_norm_squared + np.trace(sigma_true) + np.trace(sigma_artificial) - 2 * np.trace(matrix_sqrt)
    return float(fid)


def prd():
    # Precision and Recall for Distributions
    pass


if __name__ == '__main__':

    latent_true = np.random.rand(1024, 32)
    latent_art = np.random.rand(1024, 32)

    pos_fid = frechet_distance(latent_true, latent_art)
    zero_fid = frechet_distance(latent_true, latent_true)
    print(f'Positive FID = {pos_fid}')
    print(f'Zero FID = {zero_fid}')

    predictor = InceptionPredictor(output_dim=64)
    true_images = torch.rand(32, 3, 299, 299)
    art_images = torch.rand(32, 3, 299, 299)

    inc_true = predictor(true_images).cpu().numpy()
    inc_art = predictor(art_images).cpu().numpy()

    inc_fid = frechet_distance(inc_true, inc_art)
    print(f'Inception-v3 FID = {inc_fid}')
