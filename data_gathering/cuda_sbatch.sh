#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-shared
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH -t 02:00:00


$HOME/conway-game-of-life/cuda $HOME/conway-game-of-life/examples/$filename /dev/null $block_size_x $block_size_y $block_size 1000