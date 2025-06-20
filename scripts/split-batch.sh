#!/bin/bash

for val in {1..8}; do
    sbatch -J "btch-$val" commands/split-train.sh "$val"
done
