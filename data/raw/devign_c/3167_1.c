uint64_t helper_tick_get_count(void *opaque)

{

#if !defined(CONFIG_USER_ONLY)

    return cpu_tick_get_count(opaque);

#else

    return 0;

#endif

}
