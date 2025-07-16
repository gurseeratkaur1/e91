def hybrid_sort(arr):
    """
    Hybrid sorting algorithm that combines quicksort and counting sort.
    Returns the number of times counting sort is called.
    """
    count_calls = 0
    
    def counting_sort(arr, start, end):
        nonlocal count_calls
        count_calls += 1
        
        # Find the range of elements
        min_val = min(arr[start:end+1])
        max_val = max(arr[start:end+1])
        range_size = max_val - min_val + 1
        
        # Create count array
        count = [0] * range_size
        
        # Count occurrences of each element
        for i in range(start, end + 1):
            count[arr[i] - min_val] += 1
        
        # Reconstruct the sorted array
        index = start
        for i in range(range_size):
            while count[i] > 0:
                arr[index] = i + min_val
                index += 1
                count[i] -= 1
    
    def quicksort(arr, start, end):
        if start >= end:
            return
        
        # If subarray size is <= 16, use counting sort
        if end - start + 1 <= 16:
            counting_sort(arr, start, end)
            return
        
        # Choose last element as pivot
        pivot = arr[end]
        i = start - 1
        
        # Partition the array
        for j in range(start, end):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        # Place pivot in correct position
        arr[i + 1], arr[end] = arr[end], arr[i + 1]
        pivot_index = i + 1
        
        # Recursively sort left and right subarrays
        quicksort(arr, start, pivot_index - 1)
        quicksort(arr, pivot_index + 1, end)
    
    # Start the sorting process
    if len(arr) <= 16:
        counting_sort(arr, 0, len(arr) - 1)
    else:
        quicksort(arr, 0, len(arr) - 1)
    
    return count_calls

# Test with the provided sample inputs
def test_hybrid_sort():
    # Sample Testcase 1
    arr1 = [59, 89, 15, 185, 4181, 18, 154, 15, 1848, 1884, 8, 481, 8181, 84]
    arr1_copy = arr1.copy()
    result1 = hybrid_sort(arr1_copy)
    print(f"Sample Testcase 1:")
    print(f"Input: {arr1}")
    print(f"Output: {result1}")
    print(f"Sorted array: {arr1_copy}")
    print()
    
    # Sample Testcase 2
    arr2 = [55, 28, 76, 69, 77, 62, 83, 80, 28, 32, 96, 44, 61, 39, 72, 17, 8, 60, 86, 82, 22, 75, 
            41, 82, 92, 26, 44, 90, 12, 35, 72, 87, 98, 27, 1, 80, 7, 48, 63, 67, 64, 26, 30, 50, 81]
    arr2_copy = arr2.copy()
    result2 = hybrid_sort(arr2_copy)
    print(f"Sample Testcase 2:")
    print(f"Input: {arr2}")
    print(f"Output: {result2}")
    print(f"Sorted array: {arr2_copy}")

# Run the tests
test_hybrid_sort()