import torch

from dbas.gmm import GaussianMixture

@torch.no_grad()
def run_dbas(num_iters, init_data, oracle, data_min, data_max, q=0.8, n_components=5, gmm_iter=50, covariance_type='full'):
    d = init_data.shape[1]
    oracle_vals = oracle(init_data)
    gamma = torch.median(oracle_vals)
    data = init_data

    mu = None
    var = None

    for i in range(num_iters):
        try:
            model = GaussianMixture(n_components, d, covariance_type=covariance_type, mu_init=mu, var_init=var).cuda()
            model.fit(data, n_iter=gmm_iter)
            data, _ = model.sample(init_data.shape[0])
            scores = oracle(data).flatten()
            gamma = torch.quantile(scores, q)
            data = data[scores >= gamma, :]
            data = torch.clamp(data, data_min, data_max)
        except Exception as e:
            print('error in dbas at iteration', i, e)
            break
        mu = model.mu
        var = model.var
    
    return data
