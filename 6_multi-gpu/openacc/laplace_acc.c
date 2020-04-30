#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <openacc.h>
#include <omp.h>

// grid size
#define GRIDY    8192
#define GRIDX    8192

// smallest permitted change in temperature
#define MAX_TEMP_ERROR 0.02

double T_new[GRIDX+2][GRIDY+2]; // temperature grid
double T[GRIDX+2][GRIDY+2];     // temperature grid from last iteration

//   initialisation routine
void init();

int main(int argc, char *argv[]) {

    int i, j;                                            // grid indexes
    int max_iterations;                                  // maximal number of iterations
    int iteration=1;                                     // iteration
    double dt=100;                                       // largest change in temperature
    struct timeval start_time, stop_time, elapsed_time;  // timers
    int num_threads = 1;				 // number of OpenMP threads
    int thread_id  = 0;				 	 // thread ID
    int num_devices = 1;				 // number of GPU devices
    int device_id  = 0;					 // device ID
    int chunk_size = 0;					 // grid size per GPU (X direction)
    int i_start, i_end;					 // starting and ending index per GPU (X direction)
    double dt_global;

    if(argc!=2) {
      printf("Usage: %s number_of_iterations\n",argv[0]);
      exit(1);
    } else {
      max_iterations=atoi(argv[1]);
    }

    gettimeofday(&start_time,NULL); 

    init();                  

    #pragma omp parallel default(shared) firstprivate(num_threads, thread_id, num_devices, device_id, i_start, i_end, chunk_size,dt,iteration,i,j)
    {

    num_threads = omp_get_num_threads();
    thread_id  = omp_get_thread_num();

    num_devices = acc_get_num_devices(acc_device_nvidia);
    device_id  = thread_id % num_devices;
    acc_set_device_num(device_id, acc_device_nvidia);

    // calculate the chunk size based on number of threads
    chunk_size=ceil((1.0*GRIDX)/num_threads);

    // calculate boundaries and process only inner region
    i_start = thread_id * chunk_size + 1;
    i_end   = i_start + chunk_size - 1;

    // simulation iterations
    #pragma acc data copy(T), create(T_new)
    while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {

        // main computational kernel, average over neighbours in the grid
        #pragma acc kernels
        for(i = i_start; i <= i_end; i++) 
            for(j = 1; j <= GRIDY; j++) 
                T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                                            T[i][j+1] + T[i][j-1]);

        #pragma acc update self(T[i_start:1][1:GRIDY],T[i_end:1][1:GRIDY])
        #pragma omp barrier
        #pragma acc update device(T[(i_start-1):1][1:GRIDY],T[(i_end+1):1][1:GRIDY])

        // reset dt
        dt = 0.0;
        #pragma omp single
	dt_global = 0.0;
       
        #pragma omp barrier

        // compute the largest change and copy T_new to T
        #pragma acc kernels
        for(i = i_start; i <= i_end; i++){
            for(j = 1; j <= GRIDY; j++){
	      dt = fmax( fabs(T_new[i][j]-T[i][j]), dt);
	      T[i][j] = T_new[i][j];
            }
        }

        #pragma omp critical
        dt_global = fmax(dt,dt_global);

        #pragma omp barrier 

	dt=dt_global;

        // periodically print largest change
        #pragma omp master
        if((iteration % 100) == 0) 
            printf("Iteration %4.0d, dt %f\n",iteration,dt);
        
	iteration++;
    }

    }

    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); // measure time

    printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

    return 0;
}


// initialize grid and boundary conditions
void init(){

    int i,j;

    for(i = 0; i <= GRIDX+1; i++){
        for (j = 0; j <= GRIDY+1; j++){
            T[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

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
}
