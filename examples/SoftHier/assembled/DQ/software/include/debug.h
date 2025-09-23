#ifndef DEFINE_DEBUG_ONCE
#define DEFINE_DEBUG_ONCE

#if DEBUG
#include <stdio.h>
#define debug(x, ...)                                                                                                  \
    do {                                                                                                               \
        printf(x, ##__VA_ARGS__);                                                                                      \
    } while (0)
#else
#define debug(x, ...) /* x */
#endif

#endif