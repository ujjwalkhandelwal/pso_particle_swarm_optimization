# Enhanced Particle Swarm Optimization (PSO) with Python
[![GitHub license](https://img.shields.io/github/license/ujjwalkhandelwal/pso_particle_swarm_optimization)](https://github.com/ujjwalkhandelwal/pso_particle_swarm_optimization/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/ujjwalkhandelwal/pso_particle_swarm_optimization?style=flat-square
)](https://github.com/ujjwalkhandelwal/pso_particle_swarm_optimization/issues)

Implemented fully documented Particle Swarm Optimization (PSO) algorithm in Python which includes a basic model along with covering few advanced features such as **updating weight inertia, cognitive, social learning coefficients and maximum velocity of the particle**.  

## Installation
You can either download/clone this repo, or you can pip install it with the following command:
```sh
pip install git+https://github.com/ujjwalkhandelwal/pso_particle_swarm_optimization
```

## Utilities
Once the installation is finished, follow the below simple guidelines to execute PSO effectively.
```py
>>> from pso import PSO
```
Next, a fitness function (or cost function) is required. I have included **four** different fitness functions for example purposes namely `fitness_1`, `fitness_2`, `fitness_3`, and `fitness_4`.

### Fitness-1 (Himmelblau's Function)
`Minimize:` **f(x) = (x<sup>2</sup> + y - 11)<sup>2</sup> + (x + y<sup>2</sup> - 7)<sup>2</sup>**
    
`Optimum solution`:  ***x = 3 ; y = 2***

### Fitness-2 (Booth's Function)
`Minimize:` **f(x) = (x + 2y - 7)<sup>2</sup> + (2x + y - 5)<sup>2</sup>**

`Optimum solution`:  ***x = 1 ; y = 3***

### Fitness-3 (Beale's Function)
`Minimize:` **f(x) = (1.5 - x - xy)<sup>2</sup> + (2.25 - x + xy<sup>2</sup>)<sup>2</sup> + (2.625 - x + xy<sup>3</sup>)<sup>2</sup>**
    
`Optimum solution`:  ***x = 3 ; y = 0.5***

### Fitness-4
`Maximize:` **f(x) = 2xy + 2x - x<sup>2</sup> - 2y<sup>2</sup>**
    
`Optimum solution`:  ***x = 2 ; y = 1***

```py
>>> from fitness import fitness_1, fitness_2, fitness_3, fitness_4
```

Now, if you want, you can provide an initial position **X<sub>0</sub>** and bound value for all the particles (not mandatory) and optimize (minimize or maximize) the fitness function using PSO:

**NOTE:** a bool variable `min=True` (default value) for *MINIMIZATION PROBLEM* and `min=False` for *MAXIMIZATION PROBLEM*

```py
>>> PSO(fitness=fitness_1, X0=[1,1], bound=[(-4,4),(-4,4)]).execute()
```
You will see the following similar output:
```py
OPTIMUM SOLUTION
  > [3.0000078, 1.9999873]

OPTIMUM FITNESS
  > 0.0
```
When **fitness_4** is used, observe that `min=False` since it is a Maximization problem.

```py
>>> PSO(fitness=fitness_4, X0=[1,1], bound=[(-4,4),(-4,4)], min=False).execute()
```
You will see the following similar output:
```py
OPTIMUM SOLUTION
  > [2.0, 1.0]

OPTIMUM FITNESS
  > 2.0
```

Incase you want to print the fitness value for each iteration, then set `verbose=True` (here `Tmax=50` is the 
maximum iteration)

```py
>>> PSO(fitness=fitness_2, Tmax=50, verbose=True).execute()
```
You will see the following similar output:
```py
Iteration:   0  | best global fitness (cost): 18.298822
Iteration:   1  | best global fitness (cost): 1.2203953
Iteration:   2  | best global fitness (cost): 0.8178153
Iteration:   3  | best global fitness (cost): 0.5902262
Iteration:   4  | best global fitness (cost): 0.166928
Iteration:   5  | best global fitness (cost): 0.0926638
Iteration:   6  | best global fitness (cost): 0.0926638
Iteration:   7  | best global fitness (cost): 0.0114517
Iteration:   8  | best global fitness (cost): 0.0114517
Iteration:   9  | best global fitness (cost): 0.0114517
Iteration:   10 | best global fitness (cost): 0.0078867
Iteration:   11 | best global fitness (cost): 0.0078867
Iteration:   12 | best global fitness (cost): 0.0078867
Iteration:   13 | best global fitness (cost): 0.0078867
Iteration:   14 | best global fitness (cost): 0.0069544
Iteration:   15 | best global fitness (cost): 0.0063058
Iteration:   16 | best global fitness (cost): 0.0063058
Iteration:   17 | best global fitness (cost): 0.0011039
Iteration:   18 | best global fitness (cost): 0.0011039
Iteration:   19 | best global fitness (cost): 0.0011039
Iteration:   20 | best global fitness (cost): 0.0011039
Iteration:   21 | best global fitness (cost): 0.0007225
Iteration:   22 | best global fitness (cost): 0.0005875
Iteration:   23 | best global fitness (cost): 0.0001595
Iteration:   24 | best global fitness (cost): 0.0001595
Iteration:   25 | best global fitness (cost): 0.0001595
Iteration:   26 | best global fitness (cost): 0.0001595
Iteration:   27 | best global fitness (cost): 0.0001178
Iteration:   28 | best global fitness (cost): 0.0001178
Iteration:   29 | best global fitness (cost): 0.0001178
Iteration:   30 | best global fitness (cost): 0.0001178
Iteration:   31 | best global fitness (cost): 0.0001178
Iteration:   32 | best global fitness (cost): 0.0001178
Iteration:   33 | best global fitness (cost): 0.0001178
Iteration:   34 | best global fitness (cost): 0.0001178
Iteration:   35 | best global fitness (cost): 0.0001178
Iteration:   36 | best global fitness (cost): 0.0001178
Iteration:   37 | best global fitness (cost): 2.91e-05
Iteration:   38 | best global fitness (cost): 1.12e-05
Iteration:   39 | best global fitness (cost): 1.12e-05
Iteration:   40 | best global fitness (cost): 1.12e-05
Iteration:   41 | best global fitness (cost): 1.12e-05
Iteration:   42 | best global fitness (cost): 1.12e-05
Iteration:   43 | best global fitness (cost): 1.12e-05
Iteration:   44 | best global fitness (cost): 1.12e-05
Iteration:   45 | best global fitness (cost): 1.12e-05
Iteration:   46 | best global fitness (cost): 1.12e-05
Iteration:   47 | best global fitness (cost): 2.4e-06
Iteration:   48 | best global fitness (cost): 2.4e-06
Iteration:   49 | best global fitness (cost): 2.4e-06
Iteration:   50 | best global fitness (cost): 2.4e-06

OPTIMUM SOLUTION
  > [1.0004123, 2.9990281]

OPTIMUM FITNESS
  > 2.4e-06
```

Now, incase you want to plot the fitness value for each iteration, then set `plot=True` (here `Tmax=50` is the 
maximum iteration)

```py
>>> PSO(fitness=fitness_2, Tmax=50, plot=True).execute()
```
You will see the following similar output:
```py
OPTIMUM SOLUTION
  > [1.0028365, 2.9977422]

OPTIMUM FITNESS
  > 1.45e-05
```

![Fitness](https://github.com/ujjwalkhandelwal/pso_particle_swarm_optimization/blob/main/fitness.png)

Finally, in case you want to use the advanced features as mentioned above (say you want to update the weight inertia parameter `w`), simply use `update_w=True` and thats it. Similarly you can use `update_c1=True` (to update individual cognitive parameter `c1`), `update_c2=True` (to update social learning parameter `c2`), and `update_vmax=True` (to update maximum limited velocity of the particle `vmax`)

```py
>>> PSO(fitness=fitness_1, update_w=True, update_c1=True).execute()
```

### References:    

[1] Almeida, Bruno & Coppo leite, Victor. (2019). Particle swarm optimization: a powerful technique for 
solving engineering problems. 10.5772/intechopen.89633.

[2] He, Yan & Ma, Wei & Zhang, Ji. (2016). The parameters selection of pso algorithm influencing on performance of fault diagnosis. matec web of conferences. 63. 02019. 10.1051/matecconf/20166302019. 

[3] Clerc, M., and J. Kennedy. The particle swarm — explosion, stability, and convergence in a multidimensional complex space. ieee transactions on evolutionary computation 6, no. 1 (february 2002): 58–73.

[4] Y. H. Shi and R. C. Eberhart, “A modified particle swarm optimizer,” in proceedings of the ieee international
conferences on evolutionary computation, pp. 69–73, anchorage, alaska, usa, may 1998.

[5] G. Sermpinis, K. Theofilatos, A. Karathanasopoulos, E. F. Georgopoulos, & C. Dunis, Forecasting foreign exchange 
rates with adaptive neural networks using radial-basis functions and particle swarm optimization, european journal of operational research.

[6] Particle swarm optimization (pso) visually explained
(https://towardsdatascience.com/particle-swarm-optimization-visually-explained-46289eeb2e14)

[7] Rajib Kumar Bhattacharjya, Introduction to Particle Swarm Optimization 
(http://www.iitg.ac.in/rkbc/ce602/ce602/particle%20swarm%20algorithms.pdf)
