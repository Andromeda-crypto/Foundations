"""
Implementation of the silding window algorithm

Description : The Sliding Window technique is a common algorithmic approach used for solving various problems that involve
              processing or analyzing a sequential data structure such as strings, or arrays.

              It involves creating a fixed size window and moving through the data structure typically from left to right, 
              to perform specific operations or computations on the elements within the window.


              
Example question: Find the maximum sum of a subsequence of size n in the a given array arr.
                  n = 3, 
"""

def max__subarray_sum(arr,n):
    # initialize the max sum to negative infinity in case the array contains negative integers
    max_sum = float('-inf')
    # initial sum of window before iterating
    window_sum = sum(arr[:n])

    #iterating through array with fixed window size n

    for i in range(n, len(arr)):
        window_sum += arr[i] - arr[i - n] # key step to subtract the first element and update window 
        max_sum = max(max_sum, window_sum)
    
    return max_sum


