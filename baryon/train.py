import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import netket as nk
import jax
import sys
import json
from netket.utils import struct
from netket.hilbert import DiscreteHilbert

from model_ import QQQ_FCN, QQq_FCN, FCN, generate_permutations, exchange_indices
from sampler_ import GaussianFlipRule
from config_ import quarks, MASS, PARA, I, out_path, default_serializer, nlayers, nhid, bound, sigma
from hi_op_ import _nd, _nparticles, hi, hi_dis, n_discrete, ha

logpath = out_path('train')

"""State"""
if quarks[0] == quarks[1] == quarks[2]:
    @struct.dataclass
    class HilbertPyTree:
        """Internal class used to pass data."""
        hilb: DiscreteHilbert = struct.field(pytree_node=False)


    idparticles = [0, 1, 2]
    swap_indices, signs = generate_permutations(_nparticles, identical_particles=idparticles)
    model = QQQ_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), nhid=nhid, nlayers=nlayers, bound=bound,
                    hi=HilbertPyTree(hi_dis), indices=swap_indices, signs=signs)
elif quarks[0] == quarks[1]:
    idparticles = [[0, 1, (-1) ** I]]
    swap_indices, signs = exchange_indices(_nparticles, identical_particles=idparticles)
    model = QQq_FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid, nlayers=nlayers,
                    bound=bound, indices=swap_indices, signs=signs)
else:
    model = FCN(nd=_nd, nparticles=_nparticles, mass=tuple(MASS), ndiscrete=n_discrete, nhid=nhid, nlayers=nlayers,
                bound=bound, )

params = model.init(jax.random.PRNGKey(0), hi.random_state(key=jax.random.PRNGKey(0), size=1))  # 初始化参数
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
sa = nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=nchains,
                                  sweep_size=nsweep)
vs = nk.vqs.MCState(sa, model, n_samples=nsample, n_discard_per_chain=ndiscard)
op1 = nk.optimizer.Sgd(lr1)
gs = nk.VMC(ha, op1, variational_state=vs, preconditioner=sr)

gs.run(n_iter=500, out=f"{logpath}/Train1")
variables = vs.variables
train1 = {'nchains': nchains, 'nsweep': nsweep, 'nsample': nsample, 'ndiscard': ndiscard, 'lr': lr1,
          'sigma': sigma * 0.197}

"""Train 2"""
nchains, nsweep, nsample, ndiscard = 1000, 20, 1 * 10 ** 4, 200
sa = nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=nchains,
                                  sweep_size=nsweep)
vs = nk.vqs.MCState(sa, model, n_samples=nsample, n_discard_per_chain=ndiscard)
# with open(f"{logpath}/Train1.mpack", 'rb') as file:
#     variables = flax.serialization.from_bytes(vs.variables, file.read())
vs.variables = variables
train2 = {}

TRAINED = False
for i in range(10):
    lr2 = lr * (2 - i * 0.2)
    op2 = nk.optimizer.Sgd(lr2)
    gs = nk.VMC(ha, op2, variational_state=vs, preconditioner=sr)
    gs.run(n_iter=200, out=f"{logpath}/Train2_{i}")
    variables = vs.variables
    train2[i] = {'nchains': nchains, 'nsweep': nsweep, 'nsample': nsample, 'ndiscard': ndiscard, 'lr': lr2,
                 'sigma': sigma * 0.197}

    eval_vs = nk.vqs.MCState(
        nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=500,
                                     sweep_size=50), model, n_samples=5 * 10 ** 4, n_discard_per_chain=500)
    eval_vs.variables = vs.variables
    for _ in range(3):
        eval_vs.reset()
        e_ = eval_vs.expect(ha)
        print(i, e_)
        evar = e_.variance
        if evar < 0.001:
            TRAINED = True
            break
    if TRAINED:
        print("=" * 60, flush=True)
        break
if not TRAINED:
    print("Do not meet the requirance of variance")
    print("=" * 60, flush=True)

"""Eval"""
# with open(f"{logpath}/Train2.mpack", 'rb') as file:
#     variables = flax.serialization.from_bytes(vs.variables, file.read())
lr3 = lr * 0.1
op3 = nk.optimizer.Sgd(lr3)
gs = nk.VMC(ha, op3, variational_state=vs, preconditioner=sr)

nchains, nsweep, nsample, ndiscard = 500, 100, 5 * 10 ** 4, 1000
sa = nk.sampler.MetropolisSampler(hi, GaussianFlipRule(hi_dis=hi_dis, sigma=sigma), n_chains=nchains,
                                  sweep_size=nsweep)
eval_vs = nk.vqs.MCState(sa, model, n_samples=nsample, n_discard_per_chain=ndiscard)
Es = []

for i in range(5):
    gs.run(n_iter=20, out=f"{logpath}/Train3_{i}")
    eval_vs.variables = vs.variables
    Es.append(eval_vs.expect(ha))
evaluation = {'nchains': nchains, 'nsweep': nsweep, 'nsample': nsample, 'ndiscard': ndiscard, 'lr': lr3,
              'sigma': sigma * 0.197}
with open(f'{logpath}/train.json', 'w', encoding='utf-8') as f:
    json.dump({'Train1': train1, 'Train2': train2, 'Eval': evaluation}, f, indent=2, default=default_serializer,
              ensure_ascii=False)

Emean = [float(E.mean.real) for E in Es]
Evar = [float(E.variance) for E in Es]
Esigma = [float(E.error_of_mean) for E in Es]
ERhat = [float(E.R_hat) for E in Es]
results = {'Emean': Emean, 'Evar': Evar, 'Esigma': Esigma, 'ERhat': ERhat}
with open(f'{logpath}/results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, default=default_serializer, ensure_ascii=False)
