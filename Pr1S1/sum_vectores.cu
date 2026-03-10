# include <cstdio>
# include <cmath>
# include <cuda_runtime.h>

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

const size_t N[] = {1 <<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26};

static struct timeval tv0;
double getMicroSeconds()
{
    double t;
    gettimeofday(&tv0, (struct timezone*)0);
    t = ((tv0.tv_usec) + (tv0.tv_sec) * 1000000.0);
    return t;
}

int main(){


    return 0;
}