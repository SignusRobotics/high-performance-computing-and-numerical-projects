CC = gcc  
CFLAGS = -O2
LDFLAGS = -L../simple-jpeg/ -lsimplejpeg

PROJ = parallel_main
OBJS = parallel_main.o

all: simple-jpeg $(PROJ)

parallel_main: $(OBJS)
	mpicc $(CFLAGS) $< -o $@ $(LDFLAGS) -lm

parallel_main.o : parallel_main.c
	mpicc parallel_main.c -c


functions.o : functions.c functions.h
	mpicc functions.c -c

simple-jpeg:
	$(MAKE) -C ../simple-jpeg

obj-clean:
	$(RM) *.o

exec-clean:
	$(RM) $(PROJ)

autosave-clean:
	$(RM) *~

clean:
	$(MAKE) obj-clean
	$(MAKE) exec-clean
	$(MAKE) autosave-clean
	$(MAKE) -C ../simple-jpeg clean
