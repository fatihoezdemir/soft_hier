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
        if (flex_is_dm_core() && flex_get_cluster_id() == 0) {                                                         \
            flex_timer_start();                                                                                        \
        }                                                                                                              \
    } while (0)
#define PERF_END_DM()                                                                                                  \
    do {                                                                                                               \
        if (flex_is_dm_core() && flex_get_cluster_id() == 0) {                                                         \
            printf("\n\t[SYNC]");                                                                                      \
            flex_timer_end();                                                                                          \
        }                                                                                                              \
    } while (0)

// Debug print for uint16_t
#define DEBUG_PRINT_U16(ptr, count, desc)                                                                              \
    do {                                                                                                               \
        printf("[DEBUG] First %d %s:\n", (count), (desc));                                                             \
        for (int _i = 0; _i < (count); _i++) {                                                                         \
            printf("\t  [%d] = 0x%04x\n", _i, (ptr)[_i]);                                                              \
        }                                                                                                              \
    } while (0)
#if DEBUG
#define CDEBUG_PRINT_U16(ptr, count, desc) DEBUG_PRINT_U16(ptr, count, desc)
#else
#define CDEBUG_PRINT_U16(ptr, count, desc)                                                                             \
    do {                                                                                                               \
    } while (0)

#endif

#define DEBUG_PRINT_U8(ptr, count, desc)                                                                               \
    do {                                                                                                               \
        printf("[DEBUG] First %d %s:\n", (count), (desc));                                                             \
        for (int _i = 0; _i < (count); _i++) {                                                                         \
            printf("\t  [%d] = 0x%02x\n", _i, (ptr)[_i]);                                                              \
        }                                                                                                              \
    } while (0)
#if DEBUG
#define CDEBUG_PRINT_U8(ptr, count, desc) DEBUG_PRINT_U8(ptr, count, desc)
#else
#define CDEBUG_PRINT_U8(ptr, count, desc)                                                                              \
    do {                                                                                                               \
    } while (0)

#endif

#if TIMER
#define TIMER_START()                                                                                                  \
    do {                                                                                                               \
        flex_timer_start();                                                                                            \
    } while (0)
#else
#define TIMER_START()                                                                                                  \
    do {                                                                                                               \
    } while (0)
#endif
#if TIMER
#define TIMER_END()                                                                                                    \
    do {                                                                                                               \
        flex_timer_start();                                                                                            \
    } while (0)
#else
#define TIMER_END()                                                                                                    \
    do {                                                                                                               \
    } while (0)
#endif

#endif