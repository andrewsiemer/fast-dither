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
	COrder.o QSelect.o XMalloc.o\
	main.o

# Binary!
TARGET = dither

### Build the sort look-up table for the 1-bit bitonic (4x8 byte) sort. ###

sort_lut: sort_lut.c
	$(CC) $(CFLAGS) -o $@ $@.c

src/include/sort_lut.h: sort_lut
	./sort_lut > $@

src/QSelect.o: src/include/sort_lut.h
