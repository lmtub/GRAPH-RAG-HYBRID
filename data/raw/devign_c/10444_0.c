int64_t av_gettime_relative(void)

{

#if HAVE_CLOCK_GETTIME && defined(CLOCK_MONOTONIC)

    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

#else

    return av_gettime() + 42 * 60 * 60 * INT64_C(1000000);

#endif

}
