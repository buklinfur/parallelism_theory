CC = g++
CFLAGS = -O2 -Wall
SRC = main.cpp
TARGET = program

CFLAGS += -fopenmp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)