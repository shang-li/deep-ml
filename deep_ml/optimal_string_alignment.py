import numpy as np

def optimal_string_alignment(s1: str, s2: str) -> int:
    """
    ops includes: add, delete, replace, swap
    return minimum number of ops to make s1 and s2 the same 
    """
    m = len(s1)
    n = len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + np.logical_not(s1[i - 1] == s2[j - 1])
            )
            if i > 1 and j > 1 and (s1[i-2:i] == s2[j-2:j][::-1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[-1][-1]