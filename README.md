# burgers_analytical_1d_2d
Analytical burgers equation in 1d and 2d over time. 
Also contains analytical solution fundamental heat equation and fokkerplanck equation in 3d.

See `main.py` for an example on how to evaluate the solution on a spatiotemporal grid. 

Burgers implementation is based on:
Gao, Q., and M. Y. Zou. "An analytical solution for two and three dimensional nonlinear Burgers' equation." Applied Mathematical Modelling 45 (2017): 255-270.

## install
Requirements for conda are in the `.yml` file.
Else, you can just directly install the package with:
`pip install -e .` while being in the `pde_datasets` directory of this repo.

## Burgers 1d
![Alt text](plots/burgers_1d.png?raw=true "Burgers 1D")

## Burgers 2d
![Alt text](plots/burgers_2d_t0.00.png?raw=true "Burgers 2D")
![Alt text](plots/burgers_2d_t0.50.png?raw=true "Burgers 2D")
![Alt text](plots/burgers_2d_t1.00.png?raw=true "Burgers 2D")


## heat_3d

![Alt text](plots/heat_3d.png?raw=true "Burgers 2D")
