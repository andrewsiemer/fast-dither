#
# File: config.mk
# Author: Andrew Spaulding (aspauldi)
#
# Configures the specific build system for the current program.
#

# Image processing!
LIBS += -lpng -lm

# Code!
OBJECTS =\
	DTDither.o DTImage.o DTPalette.o MCQuantization.o\
	MedianPartition.o SplitImage.o XMalloc.o\
	main.o

# Binary!
TARGET = dither

# Threading? (yes/no)
USE_OMP = no

# The clang version on these machines doesn't support OpenMP. Clang took
# forever to add support, and many features are still missing, actually.
CC = gcc

### Build the sort look-up table for the 1-bit bitonic (4x8 byte) sort. ###

sort_lut: sort_lut.c
	$(CC) $(CFLAGS) -o $@ $@.c

src/include/sort_lut.h: sort_lut
	./sort_lut > $@

src/MedianPartition.o: src/include/sort_lut.h

#CC = gcc
src/MedianPartition.o: CFLAGS += -Wno-cast-align -Wno-shadow
src/MCQuantization.o: CFLAGS += -Wno-cast-align
src/DTPalette.o: CFLAGS += -Wno-cast-align
