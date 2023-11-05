import itertools
import time

def for_loops(list_of_lists):
    flattened_list = []
    for x in list_of_lists:
        for y in x:
            flattened_list.append(y)
    return flattened_list

def built_in_sum(list_of_lists):
    flattened_list = sum(list_of_lists, [])
    return flattened_list

def list_comprehension(list_of_lists):
    flattened_list = [y for x in list_of_lists for y in x]
    return flattened_list

def using_intertools(list_of_lists):
    flattened_list = list(itertools.chain(*list_of_lists))
    return flattened_list

def main():
    input = []
    for i in range(1000):
        sublist = []
        for j in range(1000):
            sublist.append(j)
        input.append(sublist)

    start = time.perf_counter()
    for_loops(input)
    end = time.perf_counter()
    print("the first method takes",end-start)

    start = time.perf_counter()
    built_in_sum(input)
    end = time.perf_counter()
    print("the second method takes",end-start)

    start = time.perf_counter()
    list_comprehension(input)
    end = time.perf_counter()
    print("the third method takes",end-start)

    start = time.perf_counter()
    using_intertools(input)
    end = time.perf_counter()
    print("the forth method takes",end-start)

if __name__ == '__main__':
    main()
