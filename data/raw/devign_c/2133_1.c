static inline int cpu_gdb_index(CPUState *cpu)

{

#if defined(CONFIG_USER_ONLY)

    return cpu->host_tid;

#else

    return cpu->cpu_index + 1;

#endif

}
