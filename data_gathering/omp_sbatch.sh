#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH -t 02:00:00


$HOME/conway-game-of-life/omp $HOME/conway-game-of-life/examples/$filename /dev/null $threads 1000