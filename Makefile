CC        := gcc
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build

TARGET    := $(BUILD_DIR)/libvindex.so

CFLAGS    := -std=c11 -Wall -Wextra -O2 -g -fPIC -I$(INC_DIR)
LDFLAGS   := -shared
LDLIBS    := -lm

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

TEST_DIR  := tests
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(TEST_SRCS))


# Each test binary links against every object file except other tests.
$(BUILD_DIR)/%: $(TEST_DIR)/%.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDLIBS)

all: $(TARGET)

test: $(TARGET) $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "--- $$t"; ./$$t || exit 1; done

.PHONY: all clean test

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d $(TARGET) $(TEST_BINS)

-include $(DEPS)