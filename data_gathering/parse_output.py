# compile with: :p you asked for how to compile stuff at the start of every file
# run with: python parse_output.py <folder> <output csv>
# a python function that opens a folder of output files from our analysis script and puts them into a csv

import os
import csv
import sys

def main():
    folder = sys.argv[1]
    output = sys.argv[2]
    files = os.listdir(folder)
    
    # open the output file
    with open(output, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, dialect='unix', quoting=csv.QUOTE_NONNUMERIC) # args mostly because git hates the excel carriage returns
        writer.writerow(['version', 'World Size', 'iterations', 'threads', 'block size x', 'block size y', 'time'])

        for file in files:
            with open(folder + '/' + file, 'r') as f:
                filename_split = file.split('-')
                filename_split[-1] = filename_split[-1][:-4] # removes the ".csv"``
                
                is_omp = len(filename_split) == 3 # worldsize-iteration-threads.csv
                is_cuda = len(filename_split) == 4 # worldsize-iteration-blocksizex-blocksizey.csv
                # serial: worldsize-iteration.csv


                version = "omp" if is_omp else "cuda" if is_cuda else "serial"
                world_size = filename_split[0]
                iterations = filename_split[1]
                threads = filename_split[2] if is_omp else 1
                block_size_x = filename_split[2] if is_cuda else 0
                block_size_y = filename_split[3] if is_cuda else 0
                time = f.readline()[5:-5].strip() # remove the "time: " and the " seconds" from the end

            writer.writerow([version, world_size, iterations, threads, block_size_x, block_size_y, time])


if __name__ == '__main__':
    main()