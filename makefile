FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: 
	nvcc -DDEBUG compute.cu nbody.cu -o nbody
clean:
	rm -f nbody 
