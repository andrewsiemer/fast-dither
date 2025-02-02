#
# File: Makefile
# Author: Andrew Spaulding (aspauldi)
#
# Builds a C-project, first into object files and then into the final target
# binary.
#
# Each object file additionally has a dependency and objdump disassembly
# generated.
#

##### Kill default build rules and set default target #####

.SUFFIXES:
default: all

##### Configure build #####

# Compiler. Clang has better warnings and static analysis.
CC := clang

# Directory containing all source code.
SRC_ROOT = src

# List of include directories.
INC = $(SRC_ROOT) $(SRC_ROOT)/include

# Libraries to link into the program.
LIBS =

# Flags for use with the compiler. Currently assumes clang.
CFLAGS =\
	-std=c99 -Wall -Werror -msse -msse2 -mavx -mavx2 -mfma -O3 -ggdb\
	$(addprefix -I,$(wildcard $(INC)))

include config.mk

# Flags which only work for clang.
CLANG_CFLAGS = -Weverything -Wno-padded -Wno-covered-switch-default\
               -Wno-unused-macros -Wno-gnu-statement-expression
ANALYZER_CFLAGS = $(CFLAGS) $(CLANG_CFLAGS) -Wno-source-uses-openmp

# Configure build for OpenMP, if requested.
ifeq ($(USE_OMP),yes)

ifeq ($(CC),gcc)
CFLAGS += -fopenmp
CLANG_CFLAGS += -Qunused-arguments -Wno-source-uses-openmp
endif
ifeq ($(CC),clang)
CFLAGS += -fopenmp -Qunused-arguments
endif

LIBS += -fopenmp
else

ifeq ($(CC),gcc)
CFLAGS += -Wno-unknown-pragmas
CLANG_CFLAGS += -Wno-source-uses-openmp
endif
ifeq ($(CC),clang)
CFLAGS += -Wno-source-uses-openmp
endif

endif

ifeq ($(CC),clang)
CFLAGS += $(CLANG_CFLAGS)
endif
ifeq ($(CC),gcc)
CFLAGS += -Wextra -Wno-strict-aliasing -Wno-missing-field-initializers
endif

# Static analysis program. We just use clang.
STATIC_ANALYSIS = clang $(ANALYZER_CFLAGS) --analyze -Xanalyzer -analyzer-output=text

# Fix the object list by making it relative to the source code root.
OBJECTS := $(addprefix $(SRC_ROOT)/,$(OBJECTS))

# Get generated file list from object list.
ASMDUMPS := $(OBJECTS:.o=.objdump.S)
DEPS := $(OBJECTS:.o=.d)

# Get the list of object files to be bulit.
ASM_OBJECTS := $(patsubst %.S,%.o,$(wildcard $(OBJECT:.o=.S)))
C_OBJECTS := $(patsubst %.c,%.o,$(wildcard $(OBJECTS:.o=.c)))

# Only include dependency files if we're not cleaning.
ifeq (0,$(words $(filter %clean,$(MAKECMDGOALS))))
-include $(OBJECTS:.o=.d)
endif

##### Build rules #####

$(ASMDUMPS): %.objdump.S: %.o
	objdump -d $^ > $@

$(ASM_OBJECTS): %.o: %.S
	$(CC) $(CFLAGS) -DASSEMBLER -c -MD -MP -MF $(@:.o=.d) -MT $@ -o $@ $(@:.o=.S)

$(C_OBJECTS): %.o: %.c
	$(STATIC_ANALYSIS) $(@:.o=.c)
ifeq ($(CC),gcc)
	clang $(CFLAGS) $(CLANG_CFLAGS) -fsyntax-only $(@:.o=.c)
endif
	$(CC) $(CFLAGS) -c -MD -MP -MF $(@:.o=.d) -MT $@ -o $@ $(@:.o=.c)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LIBS)

$(DOCS): %.1: %.pod
	pod2man --section=1 --center="$(@:.1=)" $? $@

##### Command line build targets #####

.PHONY: clean all asm target doc

all: asm target doc

asm: $(ASMDUMPS)

target: $(TARGET)

doc: $(DOCS)

clean:
	rm -f $(TARGET)
	rm -f $(OBJECTS)
	rm -f $(ASMDUMPS)
	rm -f $(DEPS)
