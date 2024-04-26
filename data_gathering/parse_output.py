# a python function that opens a folder of files and puts them into a csv

import os
import csv
import sys

def main():
    # get the folder name
    folder = sys.argv[1]
    # get the output file
    output = sys.argv[2]
    # get the list of files
    files = os.listdir(folder)
    # open the output file
    with open(output, 'w', encoding='utf-8') as csvfile:
        # create a csv writer
        writer = csv.writer(csvfile)
        # write the header
        writer.writerow(['Collected By', 'Hardware', 'serial_or_omp', 'World Size', 'iterations', 'threads', 'time'])
        # loop over the files
        for file in files:
            # open the file
            with open(folder + '/' + file, 'r') as f:
                # read the content
                collected_by = "Zach and Seth"
                hardware = "mucluster"

                filename_split = file.split('-')
                filename_split[-1] = filename_split[-1][:-4]
                is_omp = len(filename_split) == 3

                serial_or_omp = "omp" if is_omp else "serial"
                world_size = filename_split[0]
                iterations = filename_split[1]
                threads = filename_split[2] if is_omp else 1
                time = f.readline()[5:-5].strip()

            # write the content to the csv
            writer.writerow([collected_by, hardware, serial_or_omp, world_size, iterations, threads, time])


if __name__ == '__main__':
    main()