import torch

from dbas.gmm import GaussianMixture

@torch.no_grad()
def run_dbas(num_iters, init_data, oracle, data_min, data_max, q=0.8, n_components=10):
    d = init_data.shape[1]
    oracle_vals = oracle(init_data)
    gamma = torch.median(oracle_vals)
    data = init_data

    for _ in range(num_iters):
        model = GaussianMixture(n_components, d).cuda()
        model.fit(data)
        data, _ = model.sample(init_data.shape[0])
        scores = oracle(data).flatten()
        gamma = torch.quantile(scores, q)
        data = data[scores >= gamma, :]
        # truncate data
        data = torch.clamp(data, data_min, data_max)
    
    return data
        



