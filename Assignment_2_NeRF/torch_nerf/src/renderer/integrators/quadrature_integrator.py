"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from beartype import beartype as typechecker

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped(typechecker=typechecker)
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # ============================
        # Task 2. Implement Volume Rendering Equation
        # DO NOT change the code outside this part.

        sigma_delta = sigma * delta  # (num_ray, num_sample)

        alpha = 1.0 - torch.exp(-sigma_delta)  # (num_ray, num_sample)
    
        T = torch.exp(
            -torch.cat([
                torch.zeros(sigma_delta.shape[0], 1, device=sigma.device),
                torch.cumsum(sigma_delta, dim=-1)[..., :-1]
            ], dim=-1)
        )

        w_i = T * alpha  # (num_ray, num_sample)

        rgb = torch.sum(w_i.unsqueeze(-1) * radiance, dim=1)  # (num_ray, 3)
        # ============================

        return rgb, w_i
