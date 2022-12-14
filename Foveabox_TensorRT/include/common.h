#ifndef _COMMON_H
#define _COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <dirent.h>
#include "NvInfer.h"
#include <sstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>


#define CHECK_CUDA_FATAL(call)                                  \
     do {                                                        \
          cudaError_t result_ = (call);                           \
          if ( result_ != cudaSuccess )                           \
          {                                                       \
             printf(#call "failed (@loc: %d): %s \n",   \
                  __LINE__, cudaGetErrorString(result_));     \
             abort();                                            \
          }                                                       \
     } while (0)

#define CHECK_CUDA_ERROR(res_ok)                                \
    do {                                                        \
         cudaError_t status_ = cudaGetLastError();               \
         if ( status_ != cudaSuccess  )                          \
         {                                                       \
              printf("Cuda failure (@loc: %d): %s\n",    \
              __LINE__, cudaGetErrorString(status_));     \
              res_ok = false;                                     \
         }                                                       \
    } while (0)

/* CHECK the state of CUDA */
#define PERCEPTION_CUDA_CHECK(status)                                   \
    {                                                                   \
        if (status != 0)                                                \
        {                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            abort();                                                    \
        }                                                               \
    }

static Logger gLogger;

template<class T>
const T& clamp(const T& v, const T& lo, const T& hi)
{
    assert( !(hi < lo) );
    return (v < lo) ? lo : (hi < v) ? hi : v;
}


#endif