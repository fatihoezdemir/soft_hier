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
// Safe printing and timing macros - only DMA core on cluster 0 prints
#define PDEBUG(...)                                                                                                    \
    do {                                                                                                               \
        if (flex_is_dm_core() && flex_get_cluster_id() == 0)                                                           \
            printf(__VA_ARGS__);                                                                                       \
    } while (0)

#define PERF_START_DM()                                                                                                \
    do {                                                                                                               \
        if (flex_is_first_core() && flex_get_cluster_id() == 0) { /* printf("[SYNC]");  */                             \
            flex_timer_start();                                                                                        \
        }                                                                                                              \
    } while (0)
#define PERF_END_DM()                                                                                                  \
    do {                                                                                                               \
        if (flex_is_first_core() && flex_get_cluster_id() == 0)                                                        \
            flex_timer_end();                                                                                          \
    } while (0)

#endif