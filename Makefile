CC = g++
CFLAGS = -O2 -Wall
SRC = source/main.cpp
TARGET = sin_sum

ifdef USE_FLOAT
    CFLAGS += -DDATA_TYPE=float
else
    CFLAGS += -DDATA_TYPE=double
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
