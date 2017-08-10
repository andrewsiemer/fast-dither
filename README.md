# Dither - Color Quantization and Dithering

This is a small program dedicated to reducing the number of colors in an
image. It operates with files in binary PPM format ([Portable PixMap][ppm]),
and features automatic palette generation (using a median-cut algorithm) and
dithering implemented with the Floyd-Steinberg method.

## Installing

Running `make` will compile the program and generate the documentation
(requires Perl). To install, move the executable and manual file to the
appropriated directories in your system.

    $ make
    $ mv dither ~/bin
    $ mv dither.1 ~/share/man/man1

Individual targets can also be specified for only compiling or generating the
manual page:

    $ make dither
    $ make doc

## Usage

    $ dither [-p name.size] [-v] input output

Detailed information about the program options are included in the manual.

## Features

By default, `dither` will use a 3-bit RGB palette and dithering when
processing an image. Other possible palettes include:

 - Grayscale of any given size
 - Automatic, generated with the given size using the median-cut quantization
   algorithm
 - Custom palette

Dithering on the final image can be disabled and the program also can be used
to just generate a palette. For a B&W palette with no dithering, threshold
control is available.

 [ppm]: https://en.wikipedia.org/wiki/Netpbm_format
