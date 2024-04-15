import numpy as np

from task_1.func import convolve, check_signs
from src.read_utils import read_json

tests=read_json("task_1/tests.json")

filter_tensor = np.array(read_json("task_1/filters.json"))

filter_meaning = [
    "First Filter (Vertical Edges): This filter looks like it detects vertical edges.\nIf positive: increasing from right to left, not tendency: zero, if negative: increasing from left to right.",
    "Second Filter (Horizontal Edges): This filter looks like it detects horizontal edges.\nIf positive: increasing from down to top, not tendency: zero, if negative: increasing from top to down."
]

for test in tests:
    # Run convolution
    output = convolve(np.array(test["test"]), filter_tensor, stride=1, padding=0)

    for i in range(len(filter_tensor)):
        print("Input tensor:")
        print(test["test"])
        print()
        print("Filter:")
        print(filter_tensor[i])
        print()
        print("Output of the convolution by filter:")
        print(output[i])
        print()
        print("Expectations vs Received")
        print("Is sings equal: ", check_signs(np.array(test['expected'][i]), output[i]))
        print()
        print("Meaning: ")
        print(filter_meaning[i])
        print("="*100)
    
