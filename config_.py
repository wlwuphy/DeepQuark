import os
import jax.numpy as jnp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('potential', type=str, help='quark potential model, including AL1,AL2,AP1,AP2,ChQM')
parser.add_argument('quarks', type=str,
                    help='quark contents, for tetraquark the order is (QQQbarQbar), for pentaquark the order is (QQQQQbar)')
parser.add_argument('S', type=float, help='total spin')
parser.add_argument('I', type=float, help='total isospin')
parser.add_argument('--nlayers', type=int, default=None, help='number of layers')
parser.add_argument('--nnodes', type=int, default=None, help='number of hidden nodes in each layer')
parser.add_argument('--bound', type=float, default=None, help='boundary parameter (in fm)')
parser.add_argument('--sigma', type=float, default=None, help='width of gaussian distribution in Metropolis sampler')
parser.add_argument('--n', type=int, default=None, help='folder to evaluate')
parser.add_argument('--irange', type=str, default='[]', help='range to evaluate')

args = parser.parse_args()
potential, quarks, S, I = args.potential, args.quarks, args.S, args.I

defaults_dict = {
    'ccqq': {'nlayers': 4, 'nnodes': 16, 'bound': 4.0, 'sigma': 0.01},
    'bbqq': {'nlayers': 4, 'nnodes': 16, 'bound': 2.0, 'sigma': 0.01},
    'ccqqc': {'nlayers': 4, 'nnodes': 20, 'bound': 2.0, 'sigma': 0.01},
    'bbqqb': {'nlayers': 4, 'nnodes': 20, 'bound': 2.5, 'sigma': 0.01},
    'qqq': {'nlayers': 4, 'nnodes': 16, 'bound': 1.0, 'sigma': 0.01},
}

for key in ['nlayers', 'nnodes', 'bound', 'sigma']:
    if getattr(args, key) is None:
        try:
            setattr(args, key, defaults_dict[quarks][key])
        except KeyError:
            raise KeyError(f"no defaults for {quarks}")
nlayers, nhid, bound, sigma = args.nlayers, args.nnodes, args.bound / 0.197, args.sigma / 0.197
n, irange = args.n, eval(args.irange)

if 'AL1' in potential:
    masses = {'b': 5.227, 'c': 1.836, 's': 0.577, 'q': 0.315}
    PARA = {"k": 0.5069, "l": 0.1653, "L": 0.8321, "k_": 1.8609, "A": 1.6553, "B": 0.2204, "rc": 0, "p": 1}
elif potential == 'AP1':
    masses = {'b': 5.206, 'c': 1.819, 's': 0.553, 'q': 0.277}
    PARA = {"k": 0.4242, "l": 0.3898, "L": 1.1313, "k_": 1.8025, "A": 1.5296, "B": 0.3263, "rc": 0, "p": 2 / 3}
elif potential == 'AL2':
    masses = {'b': 5.231, 'c': 1.851, 's': 0.587, 'q': 0.32}
    PARA = {"k": 0.5871, "l": 0.1673, "L": 0.8182, "k_": 1.8475, "A": 1.656, "B": 0.2132, "rc": 0.1844, "p": 1}
elif potential == 'AP2':
    masses = {'b': 5.213, 'c': 1.84, 's': 0.569, 'q': 0.28}
    PARA = {"k": 0.5743, "l": 0.3978, "L": 1.1146, "k_": 1.8993, "A": 1.5321, "B": 0.3478, "rc": 0.3466, "p": 2 / 3}
elif potential == 'ChQM':
    masses = {'b': 5.11, 'c': 1.763, 's': 0.555, 'q': 0.313}
    PARA = {'mpi': 0.7 * 0.197, 'msigma': 3.42 * 0.197, 'mk': 2.51 * 0.197, 'meta': 2.77 * 0.197,
            'Lambda_pi': 4.2 * 0.197, 'Lambda_sigma': 4.2 * 0.197, 'Lambda_k': 4.21 * 0.197, 'Lambda_eta': 5.2 * 0.197,
            'gch2': 0.54 * 4 * jnp.pi, 'thetap': -15 / 180 * jnp.pi, 'alpha0': 2.118, 'Lambda0': 0.113 * 0.197,
            'mu0': 0.036976, 'r0': 0.181 / 0.197, 'ac': 0.5074, 'muc': 0.576 * 0.197, 'delta': 0.184432}
else:
    raise ValueError('Invalid potential')

MASS = []
for quark in quarks:
    MASS.append(masses[quark])

"""Output"""


def out_path(mode, n=None):
    path = f"{potential}/{''.join(quarks)}(S={S}_I={I})"
    os.makedirs(path, exist_ok=True)
    if mode == 'train':
        existing_folders = []
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            if os.path.isdir(entry_path) and entry.isdigit():
                existing_folders.append(int(entry))
        new_num = max(existing_folders) + 1 if existing_folders else 1
        logpath = os.path.join(path, str(new_num))
        os.makedirs(logpath, exist_ok=True)
    elif mode == 'eval':
        logpath = os.path.join(path, str(n))

    return logpath


def default_serializer(obj):
    if isinstance(obj, (jnp.integer, jnp.floating)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
