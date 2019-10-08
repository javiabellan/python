<h1 align="center">Python</h1>


## Theadring

Due to python GIL (global interpreter lock). A single thread can acquire that lock at a time, which means the interpreter ultimately **runs the instructions serially** ☹️. This bottleneck, however, becomes irrelevant if your program has a more severe bottleneck elsewhere, for example in network, IO, or user interaction.

Theadring is usful in:
- GUI programs: For example, in a text editing program, one thread can take care of recording the user inputs, another can be responsible for displaying the text, a third can do spell-checking, and so on.
- Network programs: For example web-scrapers. In this case, multiple threads can take care of scraping multiple webpages in parallel. The threads have to download the webpages from the Internet, and that will be the biggest bottleneck, so threading is a perfect solution here. Web servers, work similarly.

## Resources

- [Multiprocessing vs. Threading in Python](https://sumit-ghosh.com/articles/multiprocessing-vs-threading-python-data-science)
- [Python `@delegates()`](https://www.fast.ai/2019/08/06/delegation/)
- [Python Tips and Trick, You Haven't Already Seen, Part 1](https://martinheinz.dev/blog/1)
- [Python Tips and Trick, You Haven't Already Seen, Part 2](https://martinheinz.dev/blog/4)
