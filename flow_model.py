"""
Normalizing Flow (RealNVP) using nflows library.
"""

import torch
import torch.nn as nn
import numpy as np
from nflows.transforms import CompositeTransform, AffineCouplingTransform
from nflows.transforms.base import Transform
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.nn.nets import ResidualNet

class RealNVPFlow:
    def __init__(self, dim, num_layers=8, hidden_features=256, lr=1e-3, wd=1e-5, seed=42):
        self.dim = dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Define a sequence of coupling layers alternating masking patterns
        transforms = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            mask[:dim//2] = 1 if i % 2 == 0 else 0
            # For odd layers, flip mask
            # AffineCouplingTransform expects mask to be binary (1 = unchanged, 0 = transformed)
            transforms.append(AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_features, out_features:
                    ResidualNet(in_features, out_features, hidden_features=hidden_features, context_features=None,
                                num_blocks=3, activation=nn.ReLU, dropout_probability=0.0, use_batch_norm=False)
            ))
        self.transform = CompositeTransform(transforms)
        self.base_dist = StandardNormal([dim])
        self.model = Flow(self.transform, self.base_dist).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def fit(self, X, epochs=300, batch_size=128):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                loss = -self.model.log_prob(batch).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(batch)
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(X):.4f}")

    def sample(self, n_samples):
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(n_samples).cpu().numpy()
        return samples
