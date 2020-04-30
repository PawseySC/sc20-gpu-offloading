---
layout: lesson
root: .
permalink: index.html  # Is the only page that don't follow the partner /:path/index.html
---
<p align="center"><b>Organisers</b>: Tom Papatheodore (ORNL), Maciej Cytowski (PawseySC), Chris Daley (LBL)</p>

OpenACC and OpenMP are often seen as competing solutions for directive-based GPU offloading. Both models allow the programmer to offload computational workloads to run on GPUs and to manage data transfers between CPU and GPU memories. OpenACC is said to be a descriptive approach to programming GPUs, where the programmer uses directives to tell the compiler where data-independent loops are located and lets the compiler decide how/where to parallelize the loops based on the architecture (via compiler flags). OpenMP, on the other hand, is said to be a prescriptive approach to GPU programming, where the programmer uses directives to more explicitly tell the compiler how/where to parallelize the loops, instead of letting the compiler decide.
 
It’s common to hear programmers ask, “*which programming model should I use?*”, “*which approach is more portable?*”, “*are one of these models going to replace the other?*”, etc. 
<p align="center"><b>In this tutorial, we will not attempt to argue for one programming model over the other or specifically try to compare their performance profiles.</b></p> 
Instead, we will elaborate on the differences between the two approaches briefly outlined above and give participants the opportunity to explore how these differences manifest themselves in a program during hands-on exercises. We will also give a current snapshot of the compiler implementations available for the OpenACC and OpenMP (offload) specifications.

> ## Prerequisites
>
> Participants are expected to be familiar with GPU architecture, the concept of offloading computations to accelerators and one of the two discussed programming models (OpenMP or OpenACC). Participants should be familiar with C/C++ or Fortran programming language.
> Participants are required to bring their own laptops with SSH client for the hands-on session. Participants will be provided with access to the HPC platform.
{: .prereq}
