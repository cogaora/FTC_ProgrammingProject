# Program Summary
### Authors:

Lochlann Hackett - 261056757

Cian Ó Gaora - 261056892

This program builds a minimum span tree based on the edges with the highest 
reliability, and then iteratively adds new edges to the graph and uses edge 
decomposition to determine the increase in reliability to cost ratio.

The final output is a display of the final graph which meets the reliability 
target for the given cost constraint. An adjacency matrix is also printed to 
show any parallel edges between a pair of nodes


# Dependencies
### NetworkX
Install using pip in a terminal: 

`pip3 install networkx`

### Numpy
`pip3 install numpy`

### Matplotlib
`pip3 install matplotlib`

### Running the Program

As explained in the comments, the reliability target and cost constraint can be changed
at the bottom of the assignment.py file within the call to the 'driver' function.

In a terminal/cmd window, enter the project directory with the assignment.py
file and the input.txt file. Run with:

`python3 assignment.py`

