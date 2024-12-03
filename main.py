from cbmmpy import square_cube, vec_sum
import numpy as np


if __name__ == "__main__":
    print("Compute x*x + x*x*x, for x=3")
    print(f"Answer: {square_cube(3)}")

    print("Compute x + y, for x=[2. 3.] and y=[1. 1.]")
    print(f"Answer: {vec_sum(np.array([2, 3]), np.array([1, 1]))}")
