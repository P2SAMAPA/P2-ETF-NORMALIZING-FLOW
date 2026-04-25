"""
Normalizing Flow (RealNVP) using nflows library with custom MLP.
"""

import torch
import torch.nn as nn
import numpy as np
from nflows.transforms import CompositeTransform, AffineCouplingTransform
from nflows.distributions import StandardNormal
from nflows.flows import Flow

class SimpleMLP(nn.Module):
    """A simple MLP that takes input and outputs scale+shift parameters."""
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x, context=None):
        return self.net(x)

class RealNVPFlow:
    def __init__(self, dim, num_layers=8, hidden_features=256, lr=1e-3, wd=1e-5, seed=42):
        self.dim = dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)

        transforms = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            # Alternate mask: first layer transforms second half, next transforms first half
            if i % 2 == 0:
                mask[dim//2:] = 1
            else:
                mask[:dim//2] = 1

            transform_net_fn = lambda in_f, out_f: SimpleMLP(in_f, out_f, hidden_features)
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=transform_net_fn))

        self.transform = CompositeTransform(transforms)
        self.base_dist = StandardNormal([dim])
        self.model = Flow(self.transform, self.base_dist).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def fit(self, X, epochs=300, batch_size=128):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t)
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
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(X_t):.4f}")

    def sample(self, n_samples):
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(n_samples).cpu().numpy()
        return samples
