/**
 * @file XMalloc.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Malloc wrappers which panic on allocation failure.
 * @bug No known bugs.
 */

#ifndef __X_MALLOC_H__
#define __X_MALLOC_H__

#include <malloc.h>

void *XMalloc(size_t size);
void *XCalloc(size_t es, size_t nelt);
void XFree(void *p);

#endif /* __X_MALLOC_H__ */
