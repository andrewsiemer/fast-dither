/**
 * @file SplitImage.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Implementation of the split image module.
 * @bug No known bugs.
 */

#undef NDEBUG
#include <assert.h>

#include <SplitImage.h>

#include <XMalloc.h>

SplitImage *
CreateSplitImage(
    DTImage *img
) {
    assert(img);

    SplitImage *ret = XMalloc(sizeof(SplitImage));
    ret->w = img->width;
    ret->h = img->height;
    ret->type = img->type;
    ret->resolution = img->resolution;
    ret->r = XMemalign(32, sizeof(uint8_t) * ret->w * ret->h);
    ret->g = XMemalign(32, sizeof(uint8_t) * ret->w * ret->h);
    ret->b = XMemalign(32, sizeof(uint8_t) * ret->w * ret->h);

    // FIXME: Should be SIMD.
    size_t size = ret->w * ret->h;
    for (size_t i = 0; i < size; i++) {
        ret->r[i] = img->pixels[i].r;
        ret->g[i] = img->pixels[i].g;
        ret->b[i] = img->pixels[i].b;
    }

    return ret;
}

void
DestroySplitImage(
    SplitImage *img
) {
    assert(img);
    XFree(img->r);
    XFree(img->g);
    XFree(img->b);
    XFree(img);
}
