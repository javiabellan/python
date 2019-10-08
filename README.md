<h1 align="center">Python Parallelization ⏩</h1>

### Index

[todo](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9797-dask-extensions-and-new-developments-with-rapids.pdf)

- CPU
  - [**Python Threading**](#threading)
  - [**Python Multiprocessing**](#multiprocessing)
  - [**PyTorch Multiprocessing**](#pytorch-multiprocessing)
  - [**Numba JIT**](#numba)
- GPU
  - [**PyTorch CUDA**](#pytorch-cuda): Numeric parallelization similar to NumPy
  - [**Numba CUDA**](#numba-cuda): Easy parallelization
  - **cuDF**: DataFrame parallelization similar to Pandas (by RAPIDS)
  - **cuML**: Machine learn. parallelization similar to Scikit-learn (by RAPIDS)
  - **cuGraph**: graph parallelization similar to NetworkX (by RAPIDS)
  - [**CuPy**](#cupy): GPU matrix library similar to NumPy
  - **PyCuda**
  - **PyOpenCL**
  - **Dask**: Distributed parallelization

  

<h1 align="center">CPU</h1>

## Threading

Due to python GIL (global interpreter lock), only a single thread can acquire that lock at a time, which means the interpreter ultimately **runs the instructions serially** ☹️. This bottleneck, however, becomes irrelevant if your program has a more severe bottleneck elsewhere, for example in network, IO, or user interaction.

Threading is useful in:
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

#### Multiprocessing pool

```python
import multiprocessing

def f(x):
    return x*x

cores = 4
pool = multiprocessing.Pool(cores)
pool.map(f, [1, 2, 3])
```

## PyTorch (Multiprocessing)

PyTorch multiprocessing is a wrapper around the native multiprocessing module. It supports the exact same operations, but extends it, so that all tensors sent through a multiprocessing.Queue

```python
import torch.multiprocessing as mp

if __name__ == '__main__':
    num_processes = 4
    processes     = []
    for rank in range(num_processes):
        p = mp.Process(target=func, args=(x))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```

## Numba
Just-in-time (JIT) compiler for python. Works well with **loops** and **numpy**, but not with pandas

Numba also caches the functions after first use as a machine code. So after first time it will be **even faster** because it doesn’t need to compile that code again.

#### Scenarios
- **Object mode** `@jit`: Only good for checking errors with python
- **Compile mode** `@jit(nopython=True)` or also `@njit`: Good machine code performance
- **Multithreading** `@jit(nopython=True, parallel=True)`: Good if your code is parallelizable
  - Automatic multithreading of array expressions and reductions
  - Explicit multithreading of loops with `prange()`: `for i in prange(10):`
  - External multithreading with tools like concurrent.futures or Dask.
- **Vectorization SIMD** `@vectorize`
  - `@vectorize(target='cpu')`: Single-threaded CPU
  - `@vectorize(target='parallel')`: Multi-core CPU
  - `@vectorize(target='cuda')`: CUDA GPU
  
```python
from numba import jit

@jit
def function(x):
    # your loop or numerically intensive computations
    return x
    
@jit(nopython=True)
def function(a, b):
    # your loop or numerically intensive computations
    return result
    
@jit(nopython=True, parallel=True)
def function(a, b):
    # your loop or numerically intensive computations
    return result
```




<h1 align="center">GPU</h1>


## PyTorch (CUDA)
```python
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:     ", torch.cuda.get_device_name(0))
```

#### Usage
```python
tensor = torch.FloatTensor([1., 2.]).cuda()
tensor = tensor.operations ...
result = tensor.cpu()
```

#### Memory management
```python
torch.cuda.memory_allocated() # Memory usage by tensors
torch.cuda.memory_cached()    # Cache memory (visible in nvidia-smi)
torch.cuda.empty_cache()      # Free cache memory
```

## CuPy
```python
import cupy as cp
```

## Resources

- [Speed Up Your Algorithms](https://github.com/PuneetGrov3r/MediumPosts/tree/master/SpeedUpYourAlgorithms)
- [Multiprocessing vs. Threading in Python](https://sumit-ghosh.com/articles/multiprocessing-vs-threading-python-data-science)
- [Python `@delegates()`](https://www.fast.ai/2019/08/06/delegation/)
- [Python Tips and Trick, You Haven't Already Seen, Part 1](https://martinheinz.dev/blog/1)
- [Python Tips and Trick, You Haven't Already Seen, Part 2](https://martinheinz.dev/blog/4)
