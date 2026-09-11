#!/bin/bash

salloc -C gpu -q shared_interactive -t 240 -N 1 --gpus-per-node=1 --ntasks-per-node=1 -A m4539 --cpus-per-task=32 -J train
