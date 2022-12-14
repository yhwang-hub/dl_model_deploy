#!/bin/bash

mkdir -p build && cd build

cmake \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    ..

make -j4
