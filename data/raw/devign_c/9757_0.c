uint64_t cpu_get_tsc(CPUX86State *env)

{

    /* Note: when using kqemu, it is more logical to return the host TSC

       because kqemu does not trap the RDTSC instruction for

       performance reasons */

#ifdef CONFIG_KQEMU

    if (env->kqemu_enabled) {

        return cpu_get_real_ticks();

    } else

#endif

    {

        return cpu_get_ticks();

    }

}
