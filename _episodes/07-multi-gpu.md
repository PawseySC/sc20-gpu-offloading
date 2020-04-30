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
int num_threads = 1;				 // number of OpenMP threads
int thread_id  = 0;				 	 // thread ID
int num_devices = 1;				 // number of GPU devices
int device_id  = 0;					 // device ID
```
and also declared as firstprivate variables.
