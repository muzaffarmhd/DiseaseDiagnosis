import numpy as np

array = np.arange(10)
print("original array:", array)

reshaped_array = array.reshape(2, 5)
print("reshaped array (2x5):\n", reshaped_array)


ar_sum = np.sum(array)
print("sum of array elements:", ar_sum)

ar_mean = np.mean(array)
print("mean of array elements:", ar_mean)
