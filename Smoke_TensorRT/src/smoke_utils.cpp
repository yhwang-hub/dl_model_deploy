#include "../include/smoke_utils.h"

float Sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}