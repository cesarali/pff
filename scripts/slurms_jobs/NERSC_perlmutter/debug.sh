#!/bin/bash

salloc -C gpu -q debug -N 1 --gpus-per-node=4 --ntasks-per-node=4 -A m4539 --cpus-per-task=32 -J debug
