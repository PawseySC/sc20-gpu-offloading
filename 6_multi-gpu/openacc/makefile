CC=pgcc
CFLAGS=-O3 -mp -acc -Minfo=accel
LIBS=
OBJ=laplace_acc.o
TARGET=laplace_acc

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^

clean:
	rm -f $(TARGET) *.o
