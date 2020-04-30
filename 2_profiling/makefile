CC=gcc
CFLAGS=-pg -g
LIBS=
OBJ=laplace.o
TARGET=laplace

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^

clean:
	rm -f $(TARGET) *.o
