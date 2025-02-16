# Task 1. Cmake version.

### To run program use:
```
cmake -B build -DCMAKE_CXX_COMPILER=g++
cmake --build build
./build/Debug/sum_sin.exe
```
### To change data type (double/float):
By default, ```double``` is used. To change, add ```-DUSE_FLOAT=ON``` in cmake configuration line. 
For example:
```
cmake -B build -DCMAKE_CXX_COMPILER=g++ -DUSE_FLOAT=ON
```

### Output data:
* Double: ```Sum: 4.89582e-11```
* Float: ```Sum: -0.0277862```
