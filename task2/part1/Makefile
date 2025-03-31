CC = g++
CFLAGS = -O2 -Wall
SRC = main.cpp
TARGET = program

ifdef USE_BIG
    CFLAGS += -DBIG
endif

CFLAGS += -fopenmp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)