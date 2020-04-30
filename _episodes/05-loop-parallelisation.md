---
title: "Loop parallelisation"
teaching: 15
exercises: 15
questions:
- "Basic OpenACC and OpenMP directives to parallelise loop nests"
objectives:
- "Apply basic directives to parallelise loops"
- "Understand differences between OpenACC and OpenMP approaches"
keypoints:
- "We have explored differences between OpenACC and OpenMP loop constructs for GPU parallelisation"
- "We have an understanding of the difference between descriptive and prescriptive approach to GPU programming"
---

# Loop parallelisation

> ## Where to start?
> This episode starts in *3_loops/* directory. Decide if you want to work on OpenACC, OpenMP or both and follow the instructions below.  
{: .callout}

In this section we will apply basic OpenACC and OpenMP directives to parallelise loop nests identified as the most computationally expensive in the previous profiling step.

Loop parallelisation directives can be placed right before each of the loop nests in the code. There is a difference on how this can be achieved in OpenACC and OpenMP.

## OpenACC
With OpenACC programmers will usually start the parallelisation by placing the following *kernels* directive right before the first *for* loop.
```c
#pragma acc kernels
```
This directive instructs the compiler to generate parallel accelerator kernels for the loop (or loops) following the directive.

This is what we will refer to as *descriptive approach* to programming GPUs, where the programmer uses directives to tell the compiler where data-independent loops are located and lets the compiler decide how/where to parallelise the loops based on the architecture.

In the case of our Laplace example the *kernels* directive can be applied as follows:
```c
// main computational kernel, average over neighbours in the grid
#pragma acc kernels
for(i = 1; i <= GRIDX; i++)
    for(j = 1; j <= GRIDY; j++)
        T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                                    T[i][j+1] + T[i][j-1]);

// reset dt
dt = 0.0;

// compute the largest change and copy T_new to T
#pragma acc kernels
for(i = 1; i <= GRIDX; i++){
    for(j = 1; j <= GRIDY; j++){
      dt = fmax( fabs(T_new[i][j]-T[i][j]), dt);
      T[i][j] = T_new[i][j];
    }
}
```
Please note that the *kernels* directive will:
* create/destroy data environment on device(s),
* map data between host and device(s) data environment,
* attempt to parallelise loops for execution on the device,
* offload successfully parallelise loops to the target device(s),
* automatically update the data between the host and device(s).

### Important notes
1. If you are very perceptive, you might have noticed that we are cheating a little bit. We have changed
```c
      dt = MAX( fabs(T_new[i][j]-T[i][j]), dt);
```
to
```c
      dt = fmax( fabs(T_new[i][j]-T[i][j]), dt);
```
to allow the use of *fmax* intrinsic supported by the PGI compiler. This significantly improves the performance of the code.
2. If we analyse the parallel nature of the second loop nest, we can actually notice that there is a **reduction operation** that needs to be performed on *dt* variable. In this case, OpenACC can detect it automatically and apply appropriate data synchronisation technique.

## OpenMP

Similarly, with OpenMP we will start by inserting directives right before the first *for* loop of the loop nests. We will start by inserting the *target* directive, which will (for each of the structure-blocks):
* create/destroy data environment on device(s),
* map data between host and device(s) data environment,
* offloads OpenMP target regions (structured-block) to target device(s),  
* automatically update the data between the host and device(s).
Please note that compared to the OpenACC's *kernels* directive, the *target* directive will not attempt to parallelise the underlying loop nests. For this to happen, we will need to be more **prescriptive** to specify what we want to achieve.

To achieve proper parallelisation across available GPU threads we will use two following OpenMP constructs:
* *teams*, which creates a league of thread teams with the master thread of each team executing the region,
* *distribute parallel for*, which specifies a loop that can be executed in parallel by multiple threads that are members of multiple teams.

In the case of our Laplace example those directives can be applied as follows:
```c
// main computational kernel, average over neighbours in the grid
#pragma omp target
#pragma omp teams distribute parallel for collapse(2)
for(i = 1; i <= GRIDX; i++)
    for(j = 1; j <= GRIDY; j++)
        T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                                    T[i][j+1] + T[i][j-1]);

// reset dt
dt = 0.0;

// compute the largest change and copy T_new to T
#pragma omp target map(dt)
#pragma omp teams distribute parallel for collapse(2) reduction(max:dt)
for(i = 1; i <= GRIDX; i++){
    for(j = 1; j <= GRIDY; j++){
      dt = MAX( fabs(T_new[i][j]-T[i][j]), dt);
      T[i][j] = T_new[i][j];
    }
}
```
### Important notes
1. In the case of the second loop nest we are also specifying that there is a reduction on *dt* variable by adding *reduction(max:dt)* clause,
2. We are also manually specifying that variable *dt* needs to be mapped between host and device data environments. Although this will be discussed in the next step of the tutorial, for now we should just keep in mind that in OpenMP scalar variables that are not explicitly mapped are implicitly mapped as **firstprivate**.


## Comments and further analysis

Note, that in both cases we were not required to change the structure of the code to achieve GPU parallelisation. Although the Laplace example used in this tutorial gives us a space to explore various OpenACC and OpenMP directives and options, this is still a very simple program. In general cases, GPU parallelisation might require code restructure, regardless of which of the two programming paradigms is used.

**TBD** To put things another way: the kernels construct may be thought of as a hint to the compiler of where it should look for parallelism while the parallel directive is an assertion to the compiler of where there is parallelism.


> ## KEY COMMENTS
> 1. This is usually not the last step of GPU programming with directives. Deep analysis of data transfers will be done in next step. It is also important not to rely on automatic parallelisation techniques but to understand how different parameters (like block and vector sizes) might  impact the final performance.
> 2. It is really hard to judge which approach (descriptive vs prescriptive) is better. On the one hand we would like the compiler to take care of optimisations as much as possible. On the other hand programmers **must** have a clear understanding on what transformations were made to their code. **We claim that creating a highly optimised GPU code requires a very similar effort in both OpenACC and OpenMP approaches**.
{: .callout}
