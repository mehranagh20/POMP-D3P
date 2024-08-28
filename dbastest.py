from dbas.dbas import run_dbas
import torch
import time

dbas_iters = 20
dbas_q = 0.9
dbas_n_components = 5
dbas_gmm_iter = 100
num_samples = 5000
dbas_covariance_type = 'diag'
low = torch.tensor([-1., -1., -1., -1., -1., -1.]).cuda()
high = torch.tensor([1., 1., 1., 1., 1., 1.]).cuda()

oracle = lambda x: torch.norm(x, dim=1).cuda()


num_try = 3
t_sum = 0
for i in range(num_try):
    start = time.time()
    actions = torch.rand(num_samples, 6).cuda()
    action = run_dbas(dbas_iters, actions, oracle, low, high,
                        q=dbas_q, n_components=dbas_n_components, gmm_iter=dbas_gmm_iter,
                        covariance_type=dbas_covariance_type)
    t_sum += time.time() - start
                    
            
print('DBAS time:', t_sum / num_try)