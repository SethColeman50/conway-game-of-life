# to run: python make_random_grid.py <row and col>

def main():
    import random
    import csv
    import sys

    file_name = f"{sys.argv[1]}.csv"
    row = int(sys.argv[1])
    col = int(sys.argv[1])
    
    # create a grid of random 1s and 0s
    grid = []
    for i in range(row):
        row = []
        for j in range(col):
            row.append(random.choice([0, 1]))
        grid.append(row)
    
    # write the grid to a csv file
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='unix', quoting=csv.QUOTE_NONNUMERIC) # args mostly because git hates the excel carriage returns
        for row in grid:
            writer.writerow(row)

if __name__ == '__main__':
    main()