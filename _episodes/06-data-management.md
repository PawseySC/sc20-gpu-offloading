---
title: "Data management"
teaching: 15
exercises: 15
questions:
- "Usage of OpenACC and OpenMP data mapping directives"
objectives:
- "Perform basic profiling of GPU events"
- "Apply data transfer OpenACC and OpenMP directives to improve the performance of the code"
- "Understand differences between memory models"
keypoints:
- "We have successfully and significantly reduced the total number of memory transfers"
- "We have significantly increased the performance of both GPU implementations"
---

# Data management

Non-optimal memory management (e.g. excessive memory transfers) can heavily impact the performance of any GPU accelerated code. Therefore it is very important to understand how memory is being mapped and copied between host and device.  

When using PGI compiler for OpenACC this can be achieved by using **-Minfo=accel** compiler option. The information about memory transfers will be printed to stdout.

```c
pgcc -O3 -acc -Minfo=accel -c -o laplace_acc.o laplace_acc.c
main:
     43, Generating implicit copyin(T[:][:]) [if not already present]
         Generating implicit copyout(T_new[1:2048][1:2048]) [if not already present]
     44, Loop is parallelizable
     45, Loop is parallelizable
         Generating Tesla code
         44, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
         45, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
     53, Generating implicit copyin(T_new[1:2048][1:2048]) [if not already present]
         Generating implicit copy(T[1:2048][1:2048]) [if not already present]
     54, Loop is parallelizable
     55, Loop is parallelizable
         Generating Tesla code
         54, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
         55, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
         56, Generating implicit reduction(max:dt)
```

As can be seen from the above report arrays **T** and **T_new** are being copied multiple times in and out between host and device. This copying occurs in every iteration of the algorithm.

> ## Note 
> We should acknowledge the importance of *-Minfo=accel* compiler feedback option of the PGI compiler for OpenACC. GCC and Clang does not provide similar functionality for OpenMP
{: .callout}

The impact of memory transfers on the current performance of the GPU kernel can be also measured by e.g. *nvprof* profiler by running:

```bash
bash-4.2$ srun -u -n 1 nvprof ./laplace_mp 4000
```

As can be seen from the report generated below for the OpenMP version of the code, memory transfers represent more than 98% of the runtime (HtoD stands for Host to Device, DtoH stands from Device to Host).

```
==228979== Profiling application: ./laplace_mp 4000
==228979== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   51.25%  60.3494s     11236  5.3711ms  1.1520us  9.4626ms  [CUDA memcpy HtoD]
                   47.73%  56.2051s     11237  5.0018ms  1.4080us  10.967ms  [CUDA memcpy DtoH]
                    0.84%  987.39ms      2247  439.43us  430.30us  463.33us  __omp_offloading_47c4f666_4f0059e6_main_l56
                    0.18%  217.23ms      2247  96.677us  95.583us  98.144us  __omp_offloading_47c4f666_4f0059e6_main_l45
```

## Analysing data transfers

As we've seen memory transfers can take significant amount of time if scheduled improperly. In the case of the Laplace example **T** and **T_new** arrays are being copied multiple times in every iteration of the algorithm. More precisely, in each iteration of the algorithm we have:
* *T* is being copied in to the device memory (*copyin*) before the first loop nest and copied in and out of the device memory for the second loop nest (*copy*),
* *T_new* is being copied out of the device memory (*copyout*) after the first loop nest and copied in the device memory before the second loop nest (*copyin*).

This gives us 5 data transfers of a 33.5 MB buffer  per iteration and **11,000** data transfers for the entire run. However if we analyse data accesses in the implementation, we can clearly see that there is no need for this, we don't need any results on the host until after the while loop exits. We will try to fix it by using OpenACC and OpenMP compiler directives to indicate when and which data transfers should occur.

In both cases this is fairly simple. For OpenACC we place *acc data* directive right before the *while* loop:
```c
#pragma acc data copy(T), create(T_new)
while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
```
We can achieve the same for OpenMP with the use of *omp target data* directive placed right before the *while* loop:
```c
#pragma omp target data map(tofrom:T) map(alloc:T_new)
while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
```

There is actually a one to one mapping between OpenACC and OpenMP data transfer constructs.

| OpenACC construct | OpenMP construct |
| ----------------- | ---------------- |
| copyin(A)         | map(to:A)        |
| copyout(A)        | map(from:A)      |
| copy(A)           | map(tofrom:A)    |
| create(A)         | map(alloc:A)     |

Let's run the *nvprof* profiling again on the OpenMP version.
```bash
bash-4.2$ srun -u -n 1 nvprof ./laplace_mp 4000
```
```
==301161== Profiling application: ./laplace_mp 4000
==301161== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.95%  1.00671s      2247  448.03us  434.78us  458.08us  __omp_offloading_47c4f666_6901a48d_main_l56
                   16.88%  209.86ms      2247  93.396us  92.415us  95.551us  __omp_offloading_47c4f666_6901a48d_main_l45
                    1.36%  16.877ms      2250  7.5000us  1.2160us  13.957ms  [CUDA memcpy DtoH]
                    0.81%  10.120ms      2249  4.4990us  1.2790us  7.1050ms  [CUDA memcpy HtoD]
```
What we notice is that the code runs much faster now and as can be seen from the profiler information the memory transfers are taking only small fraction of runtime. GPU kernels represent around 98% of the runtime.

**We have successfully and significantly reduced the total number of memory transfers of the large *T* and *T_new* arrays: from 11,000 transfers to only 2 transfers per run.**

## Key differences

Although we claim that we have significantly reduced the number of data transfers, the *nvprof* report is still indicating that there was around 2250*2 data transfers. Those transfers are related to the use of *dt* in the second loop nest. This scalar variable needs to be copied in and out in every iteration of the algorithm. As mentioned before, there is a small difference on how the *dt* variable is declared in OpenACC and OpenMP versions of the code. In the case of OpenMP we need to be more prescriptive and specify the type of data transfers for the *dt* variable. This is related to differences in how scalar variables are treated in *kernels* and *target* constructs.


### Default scalar mapping

> ## Note 
> Scalar variables are treated slightly differently in OpenACC and OpenMP GPU regions.
{: .callout}

In OpenMP a scalar variable that is not explicitly mapped is implicitly mapped as *firstprivate*, although this behaviour can be changed with the use of *defaultmap(tofrom:scalar)* clause.

In OpenACC a scalar variables that is not explicitly mapped (copied) will be treated:
* as *firstprivate* in the parallel construct,
* as if it appeared in *copy* clause in the kernels construct.

This is why in the OpenMP implementation we need to explicitly map the *dt* variable which occurs in the *reduction* clause.

```c
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
Please be aware that similar data mapping would need to be explicitly provided if we would decide to implement OpenACC version of the code with more prescriptive *parallel* construct instead of *kernels* construct.
