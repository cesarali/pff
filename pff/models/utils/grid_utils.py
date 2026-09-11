import torch
import numpy as np

def define_mask(X,databatch):
    """
    returns mask according to the process dimensions
    Args
        X (Tensor[B,D]): 
    """
    D = X.size(1)
    B = X.size(0)
    mask = torch.arange(D, device=X.device).expand(B, -1) < databatch.process_dimension  # Shape [B, D]
    return mask

# Define Mesh Points
def define_mesh_points(total_points = 100,n_dims = 1, ranges=[])->torch.Tensor:  # Number of dimensions
    """
    returns a points form the mesh defined in the range given the list ranges
    """
    # Calculate the number of points per dimension
    number_of_points = int(np.round(total_points ** (1 / n_dims)))
    if  len(ranges) == n_dims:
    # Define the range for each dimension
        axes_grid = [torch.linspace(ranges[_][0], ranges[_][1], number_of_points) for _ in range(n_dims)]
    else:
        axes_grid = [torch.linspace(-1.0, 1.0, number_of_points) for _ in range(n_dims)]
    # Create a meshgrid for n dimensions
    meshgrids = torch.meshgrid(*axes_grid, indexing='ij')
    # Stack and reshape to get the observation points
    points = torch.stack(meshgrids, dim=-1).view(-1, n_dims)
    return points