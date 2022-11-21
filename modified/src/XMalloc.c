/**
 * @file XMalloc.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Malloc wrappers which panic on allocation failure.
 * @bug No known bugs.
 */

#include <XMalloc.h>

#undef NDEBUG
#include <assert.h>

void *
XMalloc(
    size_t size
) {
    void *ret = malloc(size);
    assert(ret);
    return ret;
}

void *
XMemalign(
    size_t align,
    size_t size
) {
    void *ret = memalign(align, size);
    assert(ret);
    return ret;
}

void *
XCalloc(
    size_t es,
    size_t nelt
) {
    void *ret = calloc(es, nelt);
    assert(ret);
    return ret;
}

void
XFree(
    void *p
) {
    free(p);
}
