# Parallel_Computing
## Project1 (Heat Transport on a 2D grid, Using MPI)
### Usage
1. Clone the repository
```markdown
  git clone https://github.com/shellswestern/Parallel_Computing.git
```
2. Run the program 
```markdown
  cd Project1
  mkdir build
  cd build
  cmake ..
  cmake --build .
  ./mpi_project
```
### Data
Input->output:

![image](https://github.com/shellswestern/Parallel_Computing/blob/master/Project1/image.png)
![image](https://github.com/shellswestern/Parallel_Computing/blob/master/Project1/result.png)


## Project2 (Matrix-Vector Product in 7 methods, Using CUDA)
### Usage
1. Clone the repository
```markdown
  git clone https://github.com/shellswestern/Parallel_Computing.git
```
2. Run the program & check the performance
```markdown
  cd Project2
  nvprof ./matVec 1 512 (example numbers)
```
