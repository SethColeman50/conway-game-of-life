# things to find crossover in
# number of threads
# number of nodes
# world size


# serial
for world_size in 8 16 32 64 128 256; do
    for iter in 1 2 3; do
        echo $world_size world_size, $iter iteration
        sbatch -o /home/bingamanz/conway-game-of-life/output/$world_size-$iter.out \
            -e /home/bingamanz/conway-game-of-life/errors/$world_size-$iter.err \
            --filename=$world_size.csv ./omp_sbatch.sh
    done
done

# omp
for threads in $(seq 1 1 8); do
    for world_size in 8 16 32 64 128 256; do
        for iter in 1 2 3; do
            echo $threads threads, $world_size world_size, $iter iteration
            sbatch -o /home/bingamanz/conway-game-of-life/output/$world_size-$iter-$threads.out \
                -e /home/bingamanz/conway-game-of-life/errors/$world_size-$iter-$threads.err \
                --export=threads=$threads,filename=$world_size.csv \
                --cpus-per-task=$threads ./omp_sbatch.sh
        done
    done
done

# cuda
