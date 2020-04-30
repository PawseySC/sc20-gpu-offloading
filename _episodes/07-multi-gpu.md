---
title: "Multi-GPU implementation"
teaching: 45
exercises: 30
questions:
- "Extra: implementing multi-GPU parallelisation"
objectives:
- "Use OpenMP to create mutliple CPU threads"
- "Use OpenACC and OpenMP API functions to assign threads to devices"
- "Use update directives to synchronise halo-exchange boundaries"
keypoints:
- "We have learned how to parallelise OpenACC and OpenMP applications on multiple GPUs within the node"
- "We are now able to create codes that use all computational devices available in the node"
---

# Mutli-GPU implementation

> ## Where to start?
> This episode starts in *5_single-gpu/* directory. Decide if you want to work on OpenACC, OpenMP or both and follow the instructions below.  
{: .callout}

## Strategies
GPU-accelerated HPC systems are usually based on multi-GPU nodes. This means that in order to take advantage of the full computational power of the node we need to parallelise our application across multiple GPUs within the node. There are at least two strategies to achieve this:
* use MPI to create multiple processes and assign processes to GPUs,
* use multithreading programming model (e.g. OpenMP) and assign threads to GPUs.

In this episode we will use OpenMP to generate multiple threads and assign threads to GPUs. Each of the threads will be assigned to its unique GPU.

The computational nature of the Laplace equation solver will require synchronisation on the boundaries of domains assigned to various GPUs.

## Multithreading and thread-GPU affinity

We will start by creating OpenMP parallel region around the *while* loop. We will set the default data type to be *shared*.  

```c
#pragma omp parallel default(shared) firstprivate(...)
{
  ...
  while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
  ...
}
```
Now that we have created multiple threads, we will create affinity between threads and GPUs. For this we will use OpenMP and OpenACC API query functions.

For this to work we must first include appropriate header files:
```c
#include <openacc.h>
#include <omp.h>
```

First, we will query for the number of available threads and set a thread ID.

```c
  num_threads = omp_get_num_threads();
  thread_id  = omp_get_thread_num();
```

Next, we will query for the number of devices and compute the GPU device ID for each thread by taking thread ID module number of available GPU devices.

```c
  num_devices = acc_get_num_devices(acc_device_nvidia);
  device_id  = thread_id % num_devices;
  acc_set_device_num(device_id, acc_device_nvidia);
```

Finally, we can assign thread to GPU with the use of language specific function. Please note that this will look differently for OpenACC and OpenMP:
* OpenACC
```c
acc_set_device_num(device_id, acc_device_nvidia);
```
* OpenMP
```c
omp_set_default_device(device_id);
```

Please note that we have used new integer variables: *num_threads, thread_id, num_devices, device_id*.

Those need to be defined earlier in the code:
```c
int num_threads = 1;				 
int thread_id  = 0;				 	 
int num_devices = 1;				 
int device_id  = 0;					 
```
and also declared as **firstprivate** variables.

## Domain decomposition

Parallelisation strategy and proper algorithm design is one the most important steps for achieving high computational performance. In the case of our Laplace example we will simply divide the problem into *num_threads* chunks (stripes) by dividing the grid in X dimension. For this we need to:
* compute the size of the chunk for each thread:
```c
// calculate the chunk size based on number of threads
chunk_size=ceil((1.0*GRIDX)/num_threads);
```
* calculate X direction loop bounds for each thread:
```c
// calculate boundaries and process only inner region
i_start = thread_id * chunk_size + 1;
i_end   = i_start + chunk_size - 1;
```
Please note that all those variable need to be defined earlier as well as declared as **firstprivate**.

```c
int chunk_size = 0;					 
int i_start, i_end;					
```

We can now change both i-loops to iterate within the precomputed boundaries:
```c
for(i = i_start; i <= i_end; i++)
```

> ## Caution
> There is one very important component missing: halo-exchange. Left-hand side and right-hand side boundaries of each stripe (chunk) need to be synchronised with other threads. Please continue reading to learn how that can be achieved with OpenACC and OpenMP.   
{: .callout}

## Firstprivate variables

We have set the CPU default OpenMP data type to *shared*. For this reason we need to be very careful with identifying variables that need to be treated as private of firstprivate within the OpenMP parallel region.

Clearly, arrays *T* and *T_new* should be treated as shared. Those arrays can be potentially very big in size and each thread will be working on a separate stripe/chunk.

We have also noted before that *num_threads, thread_id, num_devices, device_id, chunk_size, i_start* and *i_end* variables need to be treated as firstprivate.

In addition to this, variables *dt, iteration, i* and *j* need to be treated as firstprivate as well.

Therefore, the opening of the OpenMP parallel region should be of the following form:
```c
#pragma omp parallel default(shared) firstprivate(num_threads, thread_id, num_devices, device_id, i_start, i_end, chunk_size,dt,iteration,i,j)
```

## Halo-exchange

There is one very important component missing: halo-exchange. Left-hand side and right-hand side boundaries of each stripe (chunk) need to be synchronised with other threads. Please continue reading to learn how that can be achieved with OpenACC and OpenMP. This is related to the computational nature of the Laplace equation solver. Value for each grid node is computed as an average of its 4 neighbours.

It is even more complicated since GPU threads are operating on a local copy of *T* and *T_new* arrays which were transferred to the GPU memory.

> ## Question
> Do we need to transfer the whole *T* and *T_new* arrays back to CPU memory to achieve the synchronisation? We have already seen that copying whole arrays back and forth is a significant overhead.
{: .callout}

Fortunately, both OpenMP and OpenACC provide us with *update* directives which can be used to transfer only single rows or columns of 2D arrays. Therefore, each thread will:
* first copy its stripe (chunk) boundaries back to CPU memory,  
* next copy neighbouring thread boundaries from CPU memory to GPU memory.
We will place those two events right after the main computational kernel in the code.

Please note that we had to introduce an additional OpenMP barrier between those two events. We need to be sure that boundaries of neighbouring threads are already available in the CPU memory.

This can be implemented in the following way for OpenACC:
```c
// main computational kernel, average over neighbours in the grid
#pragma acc kernels
for(i = i_start; i <= i_end; i++)
    for(j = 1; j <= GRIDY; j++)
        T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                                    T[i][j+1] + T[i][j-1]);

#pragma acc update self(T[i_start:1][1:GRIDY],T[i_end:1][1:GRIDY])
#pragma omp barrier
#pragma acc update device(T[(i_start-1):1][1:GRIDY],T[(i_end+1):1][1:GRIDY])
```
Similarly, for OpenMP:
```c
// main computational kernel, average over neighbours in the grid
#pragma omp target
#pragma omp teams distribute parallel for collapse(2)
for(i = i_start; i <= i_end; i++)
    for(j = 1; j <= GRIDY; j++)
        T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                                    T[i][j+1] + T[i][j-1]);

#pragma omp target update from(T[i_start:1][1:GRIDY])
#pragma omp target update from(T[i_end:1][1:GRIDY])
#pragma omp barrier
#pragma omp target update to(T[(i_start-1):1][1:GRIDY])
#pragma omp target update to(T[(i_end+1):1][1:GRIDY])
```

## Synchronisation

Every thread is now able to compute correct result in each iteration, however we do not have proper synchronisation for the *dt* variable which is used to determine if the iterative algorithm converged. For that reason we will create a shared *dt_global* variable to compute global largest change of temperature.

This can be achieved in the following way:
```c
// reset dt
dt = 0.0;
#pragma omp single
dt_global = 0.0;

#pragma omp barrier

// compute the largest change and copy T_new to T
/*
OpenACC or OpenMP loop construct
*/
for(i = i_start; i <= i_end; i++){
    for(j = 1; j <= GRIDY; j++){
dt = MAX( fabs(T_new[i][j]-T[i][j]), dt);
T[i][j] = T_new[i][j];
    }
}

#pragma omp critical
dt_global = MAX(dt,dt_global);

#pragma omp barrier

dt=dt_global;
```
Local *dt* is first set to zero. One of the OpenMP threads is also setting *dt_global* to zero, this is followed by OpenMP barrier for proper synchronisation.

After computing local *dt*, each thread updates *dt_global*. This needs to be done in the OpenMP critical region to make sure that each thread's value is taken into account. Work is synchronised with the use of OpenMP barrier to make sure that contribution from all threads were calculated. Finally, each thread copies *dt_global* value to local *dt*.

## Diagnostic messages
We also make sure that diagnostic messages are printed by only a single thread (in that case by the master thread).

```c
// periodically print largest change
#pragma omp master
if((iteration % 100) == 0)
    printf("Iteration %4.0d, dt %f\n",iteration,dt);
```

> ## Note
> The size of the grid should be significantly increased to measure scalability across multiple GPUs. We proposed to use 8192x8192 grid size.
{: .callout}
