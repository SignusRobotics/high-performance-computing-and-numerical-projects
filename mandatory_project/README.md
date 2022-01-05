
# README IN4200_README_15362

# How to build the program

Execute makefile with clean target and then build. The build target is default, so the following command would clean and build. 

```bash
make clean && make
```

## How to run the program

### Serial code
To execute navigate to ```serial_code``` and execute ``` ./serial_main mona_lisa_noisy.jpg mona_lisa_image2.jpg 0.2 100``` 

### Parallel code
To execute navigate to ```parallel_code``` and execute ```mpirun -np 25 ./parallel_main mona_lisa_noisy.jpg mona_lisa_image2.jpg 0.2 100``` 
