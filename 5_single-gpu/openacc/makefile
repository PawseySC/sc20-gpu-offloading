CC=pgcc
F90=pgf90
CFLAGS=-O3 -acc -Minfo=accel
FFLAGS=-O3 -acc -Minfo=accel
LIBS=
OBJ=laplace.o,laplacef.o
TARGET= laplace laplacef

all: $(TARGET)

laplace.o: laplace.c
	$(CC) $(CFLAGS) -c -o $@ $<

laplacef.o: laplace.f90
	$(F90) $(FFLAGS) -c -o $@ $<

laplace: laplace.o
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^

laplacef: laplacef.o
	$(F90) $(FFLAGS) $(LIBS) -o $@ $^

clean:
	rm -f $(TARGET) *.o
