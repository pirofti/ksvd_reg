# Regularized K-SVD Algorithm

Implementation of the [Regularized K-SVD](http://ieeexplore.ieee.org/abstract/document/7831399/) Dictionary Learning Algorithm described in
B. Dumitrescu and P. Irofti, "Regularized K-SVD," in IEEE Signal Processing Letters, vol. 24, no. 3, pp. 309-313, March 2017.

## Prerequisite
[OMP](http://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip) implementation by Ron Rubinstein

## Usage
### INPUTS:
- Y -- training signals set
- D -- current dictionary
- X -- sparse representations
- iter -- current DL iteration

### PARAMETERS:
- reg -- regularization factor (default: 0.01)
- vanish -- regularization vanishing factor (default: 0.95)
- regstop -- stop regularization from this iteration on (default: Inf)

### OUTPUTS:
- D -- updated dictionary
- X -- updated representations

Sample call

    [D,X] = ksvd_reg(Y,D,X,iter)
    
to be used within a dictionary learning loop (see [DL](DL.m)).

Have a look at the [test script](test_ksvd_reg.m) for a full example.
