# change this to wite the grid with 1s and 0s and into a csv file
def main():
    import random
    import csv
    
    #parse file name from command line
    import sys
    file_name = sys.argv[1]
    
    # prase row and col from command line
    import sys
    row = int(sys.argv[2])
    col = int(sys.argv[3])
    
    # create a grid of random 1s and 0s
    grid = []
    for i in range(row):
        row = []
        for j in range(col):
            row.append(random.choice([0, 1]))
        grid.append(row)
    
    # write the grid to a csv file
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='unix', quoting=csv.QUOTE_NONNUMERIC)
        for row in grid:
            writer.writerow(row)

if __name__ == '__main__':
    main()