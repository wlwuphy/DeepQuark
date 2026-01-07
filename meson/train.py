import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
import jax
import sys
import json

from model_ import FCN
from config_ import MASS, PARA, out_path, default_serializer, nlayers, nhid, bound, sigma
from hi_op_ import _nd, _nparticles, hi, ha

logpath = out_path('train')

"""State"""

model = FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), nhid=nhid, nlayers=nlayers, bound=bound, )

params = model.init(jax.random.PRNGKey(0), hi.random_state(key=jax.random.PRNGKey(0), size=1))
layer_nodes = []
for key in params['params'].keys():
    if 'kernel' in params['params'][key]:
        output_nodes = params['params'][key]['kernel'].shape[-1]
        layer_nodes.append(output_nodes)
total_params = sum(p.size for p in jax.tree_util.tree_leaves(params['params']))

params = {
    'hyperparams': {'nodes': layer_nodes, 'para_num': total_params, 'bound': f"exp(-rij^2/{int(bound * 0.197)}^2)"},
    'potential_params': {'mass': MASS, 'para': PARA}}

with open(f'{logpath}/parameters.json', 'w', encoding='utf-8') as f:
    json.dump(params, f, indent=2, default=default_serializer, ensure_ascii=False)

"""Optimization"""
lr = 1 * 10 ** -2
sr = nk.optimizer.SR()

"""Train 1"""
nchains, nsweep, nsample, ndiscard = 1000, 20, 1 * 10 ** 4, 100
lr1 = lr * 2
sa = nk.sampler.MetropolisGaussian(hi, sigma=sigma, n_chains=nchains, sweep_size=nsweep)
vs = nk.vqs.MCState(sa, model, n_samples=nsample, n_discard_per_chain=ndiscard)
op1 = nk.optimizer.Sgd(lr1)
gs = nk.VMC(ha, op1, variational_state=vs, preconditioner=sr)

gs.run(n_iter=500, out=f"{logpath}/Train1")
variables = vs.variables
