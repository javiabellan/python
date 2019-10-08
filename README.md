<h1 align="center">Python</h1>

### Index

-  Parallelization ⏩
   - CPU
     - [Theadring](#theadring)
     - [Multiprocessing](#multiprocessing)
     - Numba
   - GPU
     - PyCuda
     - PyOpenCL
     - PyTorch
     - Dask
   

## Theadring

Due to python GIL (global interpreter lock), only a single thread can acquire that lock at a time, which means the interpreter ultimately **runs the instructions serially** ☹️. This bottleneck, however, becomes irrelevant if your program has a more severe bottleneck elsewhere, for example in network, IO, or user interaction.

Theadring is usful in:
- GUI programs: For example, in a text editing program, one thread can take care of recording the user inputs, another can be responsible for displaying the text, a third can do spell-checking, and so on.
- Network programs: For example web-scrapers. In this case, multiple threads can take care of scraping multiple webpages in parallel. The threads have to download the webpages from the Internet, and that will be the biggest bottleneck, so threading is a perfect solution here. Web servers, work similarly.

```python
import threading

def func(x):
    return x*x

thread1 = threading.Thread(target=func, args=(4))
thread2 = threading.Thread(target=func, args=(5))

thread1.start() # Starts the thread asynchronously
thread2.start() # Starts the thread asynchronously

thread1.join()  # Wait to terminate
thread2.join()  # Wait to terminate
```

## Multiprocessing

Multiprocessing outshines threading in cases where the program is CPU intensive and doesn’t have to do any IO or user interaction. For example, any program that just crunches numbers.

```python
import multiprocessing

def func(x):
    return x*x

process1 = multiprocessing.Process(target=func, args=(4))
process2 = multiprocessing.Process(target=func, args=(5))

process1.start() # Start the process
process2.start() # Start the process

process1.join()  # Wait to terminate
process2.join()  # Wait to terminate
```

### Multiprocessing pool

```python
import multiprocessing

def f(x):
    return x*x

cores = 4
pool = multiprocessing.Pool(cores)
pool.map(f, [1, 2, 3])
```


## Resources

- [Speed Up Your Algorithms](https://github.com/PuneetGrov3r/MediumPosts/tree/master/SpeedUpYourAlgorithms)
- [Multiprocessing vs. Threading in Python](https://sumit-ghosh.com/articles/multiprocessing-vs-threading-python-data-science)
- [Python `@delegates()`](https://www.fast.ai/2019/08/06/delegation/)
- [Python Tips and Trick, You Haven't Already Seen, Part 1](https://martinheinz.dev/blog/1)
- [Python Tips and Trick, You Haven't Already Seen, Part 2](https://martinheinz.dev/blog/4)
