CC = g++
CFLAGS = -O2 -Wall -lpthread
SRC = main.cpp
TARGET = program

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)