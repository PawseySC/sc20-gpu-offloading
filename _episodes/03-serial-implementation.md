---
title: "Serial Implementation"
teaching: 15
exercises: 0
questions:
- "A quick overview of the serial implementation of a 2D Laplace equation solver"
objectives:
- "Understand data structures used in the implementation"
- "Understand code structure and functions defined in the code"
keypoints:
- "2D Laplace equation solver is easy to implement. In the implementation we are using basic programming constructs like while and for loops."
---

# Serial implementation

> ## Where to start?
> This episode starts in *1_serial/* directory. Decide if you want to work on OpenACC, OpenMP or both and follow the instructions below.  
{: .callout}

## Code structure
### Data structures
The number of grid nodes in each direction of the 2D grid is defined at the top of the file as:

```c
// grid size
#define GRIDY    2048
#define GRIDX    2048
```

The main data structures used in the code are two 2D double precision arrays representing temperature grids from current and previous iterations.

```c
double T_new[GRIDX+2][GRIDY+2]; // temperature grid
double T[GRIDX+2][GRIDY+2];     // temperature grid from last iteration
```
The size of those arrays in each direction is equal to the number of grid nodes in that direction plus 2. The additional padding is used to allow for computation of boundary elements.

### Functions
There are two functions defined in the code:
* initialisation function ```init()``` which sets the initial and boundary condition,
* main function ```main(int argc, char *argv[])``` which reads code input parameters, calls the initialisation function and implements the main algorithm.   

## Initialisation function
The initialisation function sets the initial condition to 0 in every grid node.

```c
    for(i = 0; i <= GRIDX+1; i++){
        for (j = 0; j <= GRIDY+1; j++){
            T[i][j] = 0.0;
        }
    }
```

The initialisation function also sets the boundary conditions of the grid to have the value 0 for the row/column where the first/second index are zero. Along the other two boundaries the temperature will increase linearly from 0 to 128.

```c
    // set left side to 0 and right to a linear increase
    for(i = 0; i <= GRIDX+1; i++) {
        T[i][0] = 0.0;
        T[i][GRIDY+1] = (128.0/GRIDX)*i;
    }

    // set top to 0 and bottom to linear increase
    for(j = 0; j <= GRIDY+1; j++) {
        T[0][j] = 0.0;
        T[GRIDX+1][j] = (128.0/GRIDY)*j;
    }
```

## Main functions

The most important part of the main function is the implementation of Laplace equation solver. The algorithm iterations are implemented as a ```while``` loop. The ```while``` loop will terminate when one of the following conditions are satisfied:
- the number of iterations ```iteration``` exceeds the maximum number of iterations ```max_iteration``` defined by the user as a command line parameter,
- the largest change in temperature `from the previous iteration ```dt``` is lower than ```MAX_TEMP_ERROR```.

```c
// simulation iterations
while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
```

Within the ```while``` loop the main computational kernel is implemented. The new temperature for every grid node is computed as an average over neighbours in the grid.  

```c

    // main computational kernel, average over neighbours in the grid
    for(i = 1; i <= GRIDX; i++)
        for(j = 1; j <= GRIDY; j++)
            T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                                        T[i][j+1] + T[i][j-1]);
```
Next, the largest change in the temperature is computed and ```dt``` is updated. ```T_new``` is copied to ```T``` to allow for update in the next iteration. Diagnostic information is periodically printed (every 100 iterations) to monitor the convergence of the iterative algorithm.

```c
    // reset dt
    dt = 0.0;

    // compute the largest change and copy T_new to T
    for(i = 1; i <= GRIDX; i++){
        for(j = 1; j <= GRIDY; j++){
          dt = MAX( fabs(T_new[i][j]-T[i][j]), dt);
          T[i][j] = T_new[i][j];
        }
    }

    // periodically print largest change
    if((iteration % 100) == 0)
        printf("Iteration %4.0d, dt %f\n",iteration,dt);

    iteration++;
```
## Time measurements

The execution time of the algorithm is measured and reported with the use of *gettimeofday* function.

```c
gettimeofday(&start_time,NULL);
...    
gettimeofday(&stop_time,NULL);
timersub(&stop_time, &start_time, &elapsed_time); // measure time

printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
```
