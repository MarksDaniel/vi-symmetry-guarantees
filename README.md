# Symmetry Guarantees Statistic Recovery<br>in Variational Inference

This repository contains the official implementation of the examples in the NeurIPS 2026 paper **[Symmetry Guarantees Statistic Recovery in Variational Inference](https://arxiv.org/abs/2604.18310)** by Daniel Marks, Dario Paccagnan, and Mark van der Wilk.

## Reproducing the Figures

To reproduce the environment, use Python 3.9 and install the pinned dependencies in `requirements.txt`.

```bash
git clone https://github.com/MarksDaniel/vi-symmetry-guarantees.git
cd vi-symmetry-guarantees

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To reproduce the figures, run the following commands with the virtual environment activated:

```bash
python even_elliptical_symmetry_vi.py
python rotational_symmetry_vi.py
```

The scripts create an `outputs/` directory and save the figures as PDFs. Each script can be run independently.

| Figure | Example | Output |
| --- | --- | --- |
| Fig. 1 | Gaussian VI under even and elliptical symmetry | `outputs/vi_even_elliptical_example.pdf` |
| Fig. 2 | Fixed-concentration von Mises–Fisher VI under rotational symmetry | `outputs/vi_sphere_example.pdf` |
