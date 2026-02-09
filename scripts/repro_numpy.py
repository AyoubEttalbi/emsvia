import numpy as np

def test():
    print("Testing NumPy ambiguity...")
    
    # Case 1: if array
    try:
        arr = np.array([1, 2])
        if arr:
            print("Safe?")
    except Exception as e:
        print(f"Case 1 (if arr): {e}")

    # Case 2: if arr is not None
    try:
        arr = np.array([1, 2])
        if arr is not None:
            print("Case 2 (if arr is not None) is SAFE")
    except Exception as e:
        print(f"Case 2: {e}")

    # Case 3: if scalar != scalar
    try:
        s1 = np.int64(1)
        s2 = np.int64(2)
        if s1 != s2:
            print("Case 3 (id != id) is SAFE")
    except Exception as e:
        print(f"Case 3: {e}")

    # Case 4: if list != list
    try:
        l1 = [1, 2]
        l2 = [3, 4]
        if l1 != l2:
            print("Case 4 (list != list) is SAFE")
    except Exception as e:
        print(f"Case 4: {e}")

    # Case 5: if arr != None (NOT is None)
    try:
        arr = np.array([1, 2])
        if arr != None:
            print("Safe?")
    except Exception as e:
        print(f"Case 5 (arr != None): {e}")

    # Case 6: if arr != scalar
    try:
        arr = np.array([1, 2])
        if arr != 1:
            print("Safe?")
    except Exception as e:
        print(f"Case 6 (arr != 1): {e}")

if __name__ == "__main__":
    test()
