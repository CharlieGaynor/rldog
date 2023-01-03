import multiprocessing
import multiprocessing.pool


def step(i):
    # function that returns a list of floats
    return [1, 2]


def parallel_step(num_processes):
    with multiprocessing.pool.Pool(num_processes) as p:
        result = p.map(step, range(4))
        p.terminate()
    return result


if __name__ == "__main__":
    results = parallel_step(61)
    print(results)
    print([episode[0] for episode in results])
