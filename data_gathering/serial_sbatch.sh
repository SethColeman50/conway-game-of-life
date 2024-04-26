#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -D /home/bingamanz/conway-game-of-life

/home/bingamanz/conway-game-of-life/serial /home/bingamanz/conway-game-of-life/examples/$filename /dev/null 1000