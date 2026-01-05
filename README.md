# DeepQuark
Neural-network algorithms for multiquark bound states

Reference: https://doi.org/10.48550/arXiv.2506.20555

## Requirements
The codes are developed based on [NetKet v3.17](https://www.netket.org/). A stable package version to run this code:
- Python >= 3.12
- [JAX](https://github.com/jax-ml/jax) == 0.5.3
- NetKet == 3.17
See the [NetKet installation guide](https://netket.readthedocs.io/en/latest/install.html) for detailed instructions.

## Example Usage
```bash
python multiquark/train.py AL1 ccqq 1 0  # potential quarks S I
```
The above code carries out the NN-VMC training of the doubly charmed tetraquark system $cc\bar q\bar q$ with total spin $S=1$ and isospin $I=0$ in the AL1 potential model.
