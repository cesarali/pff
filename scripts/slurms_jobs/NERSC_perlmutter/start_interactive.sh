#!/bin/bash

salloc -C gpu -q interactive -t 240 -N $1 --gpus-per-node=4 --ntasks-per-node=4 -A m4539 --cpus-per-task=32 -J train
