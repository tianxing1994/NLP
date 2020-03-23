while True:
    n = int(input())
    array = set()
    for i in range(n):
        array.add(int(input()))

    array = sorted(list(array))
    for i in range(len(array)):
        print(array[i])

