#============================================================================
# Author:	Aaron Stange

# Summary: This project performs timing experiments for different search
# algorithms (linear, binary, and exponential) on lists of varying sizes.
# It measures the time taken for each search algorithm to find a target
# value in the list and writes the results to a CSV file named "searchTimes.csv".
# The CSV file contains the list size and the corresponding search times for
# each algorithm. Additionally, it generates a plot to visualize the performance
# of the search algorithms.

# INPUT: None

# OUTPUT: Generates a CSV file named "searchTimes.csv" containing the list size
# and the corresponding search times for each algorithm, as well as, a plot named
# "search_performance.png" to visualize the performance of the search algorithms.

# Date Last Modified:
#	 (aws) 02/07/2025 -- function comments
#============================================================================

import timeit
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
'''
import cProfile
import pstats
'''

#-----------------------------\
# linearSearch()               \
#----------------------------------------------------------------
# Accepts a list of integers (arr) and an integer (target). It performs
# a linear search on the list to find the target value.

# IN: list of integers (arr), integer (target)

# RETURNS: integer (index of target) or None if target is not found
#----------------------------------------------------------------
def linearSearch(arr: List[int], target: int) -> Optional[int]:
    """Performs linear search on a list of integers (arr) for an int (target)."""

    # loops through the list and returns the index of the target if found
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return None

#-----------------------------\
# binarySearch()               \
#----------------------------------------------------------------
# Accepts a sorted list of integers (arr) and an integer (target).
# It performs a binary search on the list to find the target value.

# IN: Sorted list of integers (arr), integer (target)

# RETURNS: integer (index of target) or None if target is not found
#----------------------------------------------------------------
def binarySearch(arr: List[int], target: int) -> Optional[int]:
    """Performs binary search on a sorted list of integers (arr) for an int (target)."""

    # sets the low and high values to the first and last index of the list
    low = 0
    high = len(arr) - 1

    # loops through the list and returns the index of the target if found
    while low <= high:
        # calculates the middle index
        mid = (low + high) // 2
        # checks if the mid value is the target and returns the index if true
        if arr[mid] == target:
            return mid
        # if the mid value is less than the target and adjusts the low value
        elif arr[mid] < target:
            low = mid + 1
        # if the mid value is greater than the target and adjusts the high value
        else:
            high = mid - 1
    return None

#-----------------------------\
# exponentialSearch()          \
#----------------------------------------------------------------
# Accepts a sorted list of integers (arr) and an integer (target).
# It performs an exponential search on the list to find the section
# where the target value is located and then performs a binary search
# on that section to find the target value.

# IN: Sorted list of integers (arr), integer (target)

# RETURNS: integer (index of target) or None if target is not found
def exponentialSearch(arr: List[int], target: int) -> Optional[int]:
    """Performs exponential search on a sorted list of integers (arr) for an int (target)."""

    # checks if the first value is the target
    if arr[0] == target:
        return 0
    i = 1
    # loops through the list and doubles the index until the target is found if there
    while i < len(arr):
        # checks if the index value is the target
        if arr[i] == target:
            return i
        i *= 2
        # if the element at the index is greater than the target
        # performs binary search on the sublist
        if arr[i] <= target:
            return binarySearch(arr[:min(i, len(arr))], target)
    return None

#-----------------------------\
# runExperiment()              \
#----------------------------------------------------------------
# Runs timing experiments for different search algorithms (linear, binary,
# and exponential) on lists of varying sizes. It measures the time taken
# for each search algorithm to find a target value in the list and writes
# the results to a CSV file named "searchTimes.csv". The CSV file contains
# the list size and the corresponding search times for each algorithm.

# IN: None

# RETURNS: None
#----------------------------------------------------------------
def runExperiment() -> None:
    """Runs timing experiments for different search algorithms and writes results to a CSV file."""

    # creates a CSV file called searchTimes.csv and writes the header
    CSV = open("searchTimes.csv", "w")
    CSV.write("List Size, Linear Search(sec), Binary Search(sec), Exponential Search(sec)\n")

    numTrials = 10

    SETUP1 = '''
range = 2**31 - 1
size = 10**3
from __main__ import linearSearch, binarySearch, exponentialSearch
from random import randint
import numpy as np
import cProfile
import pstats

# creates a random array of integers and a random target value
arr = np.random.randint(0, range, size).tolist()
target = np.random.choice(arr)

# sorts the array for binary and exponential search
sortedArr = sorted(arr)
    '''
    SETUP2 = '''
range = 2**31 - 1
size = 10**4
from __main__ import linearSearch, binarySearch, exponentialSearch
from random import randint
import numpy as np
import cProfile
import pstats

# creates a random array of integers and a random target value
arr = np.random.randint(0, range, size).tolist()
target = np.random.choice(arr)

# sorts the array for binary and exponential search
sortedArr = sorted(arr)
    '''
    SETUP3 = '''
range = 2**31 - 1
size = 10**5
from __main__ import linearSearch, binarySearch, exponentialSearch
from random import randint
import numpy as np
import cProfile
import pstats

# creates a random array of integers and a random target value
arr = np.random.randint(0, range, size).tolist()
target = np.random.choice(arr)

# sorts the array for binary and exponential search
sortedArr = sorted(arr)
    '''
    SETUP4 = '''
range = 2**31 - 1
size = 10**6
from __main__ import linearSearch, binarySearch, exponentialSearch
from random import randint
import numpy as np
import cProfile
import pstats

# creates a random array of integers and a random target value
arr = np.random.randint(0, range, size).tolist()
target = np.random.choice(arr)

# sorts the array for binary and exponential search
sortedArr = sorted(arr)
    '''
    
    setups = [SETUP1, SETUP2, SETUP3, SETUP4]
    sizes = [10**3, 10**4, 10**5, 10**6]

    for i in range(len(setups)):
        setup = setups[i]
        # times the linear, binary, and exponential search functions
        linearTime = min(timeit.repeat(stmt = "linearSearch(arr, target)", setup = setup, number = 1, repeat = numTrials))
        binaryTime = min(timeit.repeat(stmt = "binarySearch(sortedArr, target)", setup = setup, number = 1, repeat = numTrials))
        expTime = min(timeit.repeat(stmt = "exponentialSearch(sortedArr, target)", setup = setup, number = 1, repeat = numTrials))


        # Get the correct size from setupSizes
        size = sizes[i]

        # writes the results to the CSV file
        CSV.write("%d,%e,%e,%e\n" % (size, linearTime, binaryTime, expTime))

    # closes the CSV file
    CSV.close()




#--------\
# main()  \
#-------------------------------------------
def main():
    runExperiment()

    ### CODE MADE BY COPILOT, COMMENTED BY AARON STANGE ###
    # Reads our .CSV file which contains the data to be graphed
    df = pd.read_csv("searchTimes.csv")

    # Plots the data for our 3 search algorithms

    # Create 10:6 a plot
    plt.figure(figsize=(10, 6))
    # first argument is the x-axis data points(size of the list), 
    # second is the y-axis(times for search algoritms),
    # thrid is the label is for said line(name of algorithm)
    plt.plot(df["List Size"], df[" Linear Search(sec)"], label="Linear Search")
    plt.plot(df["List Size"], df[" Binary Search(sec)"], label="Binary Search")
    plt.plot(df["List Size"], df[" Exponential Search(sec)"], label="Exponential Search")

    # Add titles/labels and scales our graph(makes it all fancy)
    plt.title("Search Algorithm Performance")
    plt.xlabel("List Size")
    plt.ylabel("Time (seconds)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10**3, 10**6)
    plt.legend()
    plt.grid(True)

    # code to profile, took too long to run even when doing 1 trial(stopped it at about 5000 seconds)
    '''
    # Saves the plot as an image file and displays it
    plt.savefig("search_performance.png")
    plt.show()

    # Profile the main function
    cProfile.run('main()', 'timing_profile')

    # Create a Stats object
    p = pstats.Stats('timing_profile')

    # Sort the statistics by cumulative time and print the top 3 functions
    p.sort_stats('cumulative').print_stats(3)
    '''

# end main()

#-----------\
# START HERE \
#-----------------------------------------------------------
if (__name__ == '__main__'):
    main()

#-----------------------------------------------------
