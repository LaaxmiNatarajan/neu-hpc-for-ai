CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -Iinclude
LDFLAGS = -lpthread -lm

SRCS = src/single_dim_matrix.c src/multi_dim_matrix.c
OBJS = $(SRCS:.c=.o)

all: bin/test_matrix bin/bench_matrix

bin/test_matrix: tests/test_matrix.c $(OBJS)
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

bin/bench_matrix: bench/bench_matrix.c $(OBJS)
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf src/*.o bin
