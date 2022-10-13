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
	COrder.o QSelect.o\
	main.o

# Binary!
TARGET = dither
