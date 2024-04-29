# things to find crossover in
# number of threads
# number of nodes
# world size

# serial
for world_size in 8 16 32 64 128 256; do
    for iter in 1 2 3; do
        echo $world_size world_size, $iter iteration
        sbatch -o $HOME/conway-game-of-life/output/$world_size-$iter.out \
            -e $HOME/conway-game-of-life/errors/$world_size-$iter.err \
            --export=filename=$world_size.csv $HOME/conway-game-of-life/data_gathering/serial_sbatch.sh
    done
done

# omp
for threads in 1 2 3 4 5 6 7 8; do
    for world_size in 8 16 32 64 128 256; do
        for iter in 1 2 3; do
            echo $threads threads, $world_size world_size, $iter iteration
            sbatch -o $HOME/conway-game-of-life/output/$world_size-$iter-$threads.out \
                -e $HOME/conway-game-of-life/errors/$world_size-$iter-$threads.err \
                --export=threads=$threads,filename=$world_size.csv \
                --cpus-per-task=$threads $HOME/conway-game-of-life/data_gathering/omp_sbatch.sh
        done
    done
done

# cuda
for block_size_x in 8 16 24 32 64; do
    for block_size_y in 8 16 24 32 64; do
        for world_size in 8 16 32 64 128 256; do
            for iter in 1 2 3; do
                echo $block_size_x block_size_x, $block_size_y block_size_y, $world_size world_size, $iter iteration
                sbatch -o $HOME/conway-game-of-life/output/$world_size-$iter-$block_size_x-$block_size_y.out \
                    -e $HOME/conway-game-of-life/errors/$world_size-$iter-$block_size_x-$block_size_y.err \
                    --export=block_size_x=$block_size_x,block_size_y=$block_size_y,filename=$world_size.csv $HOME/conway-game-of-life/data_gathering/cuda_sbatch.sh
            done
        done
    done
done