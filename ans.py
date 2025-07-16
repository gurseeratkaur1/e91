def solve_monkey_array(arr):
    n = len(arr)
    def get_possible_values(x):
        values = set()
        current = x
        values.add(current)
        for _ in range(100):
            sqrt_val = int(current ** 0.5)
            if sqrt_val * sqrt_val <= current:
                values.add(sqrt_val)
                current = sqrt_val
            else:
                break
        current = x
        for _ in range(10):
            if current > 10**9:
                break
            current = current * current
            values.add(current)
        return sorted(list(values))
    possible_values = []
    for i in range(n):
        possible_values.append(get_possible_values(arr[i]))
    def min_operations(original, target):
        if original == target:
            return 0
        from collections import deque
        queue = deque([(original, 0)])
        visited = set([original])
        while queue:
            current, ops = queue.popleft()
            if current == target:
                return ops
            if ops > 20:
                continue
            sqrt_val = int(current ** 0.5)
            if sqrt_val not in visited and sqrt_val <= target * 2:
                visited.add(sqrt_val)
                queue.append((sqrt_val, ops + 1))
            if current <= 10**6:
                square_val = current * current
                if square_val not in visited and square_val <= target * 2:
                    visited.add(square_val)
                    queue.append((square_val, ops + 1))
        return float('inf')
    dp = {}
    for j, val in enumerate(possible_values[0]):
        ops = min_operations(arr[0], val)
        if ops != float('inf'):
            dp[(0, j)] = ops
    for i in range(1, n):
        new_dp = {}
        for j, val in enumerate(possible_values[i]):
            ops_to_val = min_operations(arr[i], val)
            if ops_to_val == float('inf'):
                continue
            min_cost = float('inf')
            for k, prev_val in enumerate(possible_values[i-1]):
                if prev_val <= val and (i-1, k) in dp:
                    min_cost = min(min_cost, dp[(i-1, k)] + ops_to_val)
            if min_cost != float('inf'):
                new_dp[(i, j)] = min_cost
        dp.update(new_dp)
    result = float('inf')
    for j in range(len(possible_values[-1])):
        if (n-1, j) in dp:
            result = min(result, dp[(n-1, j)])
    return result if result != float('inf') else -1

def test_examples():
    arr1 = [4, 2, 1]
    result1 = solve_monkey_array(arr1)
    print(f"Test 1: {arr1} -> {result1}")
    arr2 = [1, 3, 9, 9]
    result2 = solve_monkey_array(arr2)
    print(f"Test 2: {arr2} -> {result2}")

test_examples()

def solve_monkey_array_optimized(arr):
    n = len(arr)
    def get_reachable_values(x):
        values = {x: 0}
        current = x
        ops = 0
        while current > 1:
            sqrt_val = int(current ** 0.5)
            if sqrt_val * sqrt_val > current:
                sqrt_val -= 1
            if sqrt_val == current:
                break
            current = sqrt_val
            ops += 1
            if current not in values or values[current] > ops:
                values[current] = ops
        current = x
        ops = 0
        while current <= 10**6 and ops < 5:
            current = current * current
            ops += 1
            if current not in values or values[current] > ops:
                values[current] = ops
        return values
    all_values = []
    for i in range(n):
        all_values.append(get_reachable_values(arr[i]))
    dp = all_values[0].copy()
    for i in range(1, n):
        new_dp = {}
        for val, ops in all_values[i].items():
            min_cost = float('inf')
            for prev_val, prev_ops in dp.items():
                if prev_val <= val:
                    min_cost = min(min_cost, prev_ops + ops)
            if min_cost != float('inf'):
                new_dp[val] = min_cost
        dp = new_dp
    return min(dp.values()) if dp else -1

print("\nOptimized version:")
print(f"Test 1: {solve_monkey_array_optimized([4, 2, 1])}")
print(f"Test 2: {solve_monkey_array_optimized([1, 3, 9, 9])}")