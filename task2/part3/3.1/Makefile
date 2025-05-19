CC = g++
CFLAGS = -O2 -Wall
SRC = main.cpp
TARGET = program

CFLAGS += -fopenmp

ifdef USE_BIG
    CFLAGS += -DBIG
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)