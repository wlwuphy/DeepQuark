import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
import jax
import json

from model_ import QQqq_FCN, Q1Q2qq_FCN, exchange_indices, Penta_FCN
from sampler_ import GaussianFlipRule
from config_ import quarks, MASS, PARA, I, default_serializer, out_path, nlayers, nhid, bound, sigma
from hi_op_ import _nd, _nparticles, hi, hi_dis, n_discrete, ha

logpath = out_path('train')

"""Model"""
if quarks[0] == quarks[1] and quarks[2] == quarks[3]:
    idparticles = [[0, 1, -1], [2, 3, (-1) ** I]]
elif quarks[2] == quarks[3]:
    idparticles = [[2, 3, (-1) ** I]]
else:
    raise ValueError("Not implemented case of identical particles")

swap_indices, signs = exchange_indices(_nparticles, identical_particles=idparticles)

if _nparticles == 4:
    if quarks[0] == quarks[1]:
        model = QQqq_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid,
                         nlayers=nlayers,
                         bound=bound, indices=swap_indices, signs=signs)
    else:
        model = Q1Q2qq_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid,
                           nlayers=nlayers,
                           bound=bound, indices=swap_indices, signs=signs)
elif _nparticles == 5:
    model = Penta_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid,
                      nlayers=nlayers, bound=bound, indices=swap_indices, signs=signs)

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
# sr = nk.optimizer.SR(solver=nk.optimizer.solver.pinv_smooth)

"""Train 1"""
nchains, nsweep, nsample, ndiscard = 2000, 20, 2 * 10 ** 4, 100
lr1 = lr * 2
sa = nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=nchains,
                                  sweep_size=nsweep)
vs = nk.vqs.MCState(sa, model, n_samples=nsample, n_discard_per_chain=ndiscard)
op1 = nk.optimizer.Sgd(lr1)
gs = nk.VMC(ha, op1, variational_state=vs, preconditioner=sr)

gs.run(n_iter=1000, out=f"{logpath}/Train1")
variables = vs.variables
train1 = {'nchains': nchains, 'nsweep': nsweep, 'nsample': nsample, 'ndiscard': ndiscard, 'lr': lr1,
          'sigma': sigma * 0.197}

"""Train 2"""
nchains, nsweep, nsample, ndiscard = 2000, 20, 2 * 10 ** 4, 200
sa = nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=nchains,
                                  sweep_size=nsweep)
vs = nk.vqs.MCState(sa, model, n_samples=nsample, n_discard_per_chain=ndiscard)
vs.variables = variables
train2 = {}

TRAINED = False
for i in range(10):
    lr2 = lr * (2 - i * 0.2)
    op2 = nk.optimizer.Sgd(lr2)
    gs = nk.VMC(ha, op2, variational_state=vs, preconditioner=sr)
    gs.run(n_iter=500, out=f"{logpath}/Train2_{i}")
    variables = vs.variables
    train2[i] = {'nchains': nchains, 'nsweep': nsweep, 'nsample': nsample, 'ndiscard': ndiscard, 'lr': lr2,
                 'sigma': sigma * 0.197}

    eval_vs = nk.vqs.MCState(
        nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=2000, sweep_size=50),
        model, n_samples=4 * 10 ** 5, n_discard_per_chain=500, chunk_size=nsample // 4)
    eval_vs.variables = vs.variables
    for _ in range(5):
        eval_vs.reset()
        e_ = eval_vs.expect(ha)
        print(i, e_)
        evar = e_.variance
        if evar < 0.002:
            TRAINED = True
            break
    if TRAINED:
        print("=" * 60, flush=True)
        break
if not TRAINED:
    print("Do not meet the requirance of variance")
    print("=" * 60, flush=True)

"""Train 3 for Eval"""
# with open(f"{logpath}/Train2.mpack", 'rb') as file:
#     variables = flax.serialization.from_bytes(vs.variables, file.read())
lr3 = lr * 0.1
op3 = nk.optimizer.Sgd(lr3)
gs = nk.VMC(ha, op3, variational_state=vs, preconditioner=sr)

for i in range(10):
    gs.run(n_iter=50, out=f"{logpath}/Train3_{i}")

train3 = {'nchains': nchains, 'nsweep': nsweep, 'nsample': nsample, 'ndiscard': ndiscard, 'lr': lr3,
          'sigma': sigma * 0.197}
with open(f'{logpath}/train.json', 'w', encoding='utf-8') as f:
    json.dump({'Train1': train1, 'Train2': train2, 'Train3': train3}, f, indent=2, default=default_serializer,
              ensure_ascii=False)
