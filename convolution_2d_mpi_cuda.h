#ifndef HEADERFILE_H
#define HEADERFILE_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void compute_gpu(int *hostPartInImage, int *hostPartOutImage, int imageWidth, int imageHeight, int arg);

#endif
