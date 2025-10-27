static int get_logical_cpus(AVCodecContext *avctx)

{

    int ret, nb_cpus = 1;

#if HAVE_SCHED_GETAFFINITY && defined(CPU_COUNT)

    cpu_set_t cpuset;



    CPU_ZERO(&cpuset);



    ret = sched_getaffinity(0, sizeof(cpuset), &cpuset);

    if (!ret) {

        nb_cpus = CPU_COUNT(&cpuset);

    }

#elif HAVE_GETSYSTEMINFO

    SYSTEM_INFO sysinfo;

    GetSystemInfo(&sysinfo);

    nb_cpus = sysinfo.dwNumberOfProcessors;

#elif HAVE_SYSCTL && defined(HW_NCPU)

    int mib[2] = { CTL_HW, HW_NCPU };

    size_t len = sizeof(nb_cpus);



    ret = sysctl(mib, 2, &nb_cpus, &len, NULL, 0);

    if (ret == -1)

        nb_cpus = 0;

#elif HAVE_SYSCONF && defined(_SC_NPROC_ONLN)

    nb_cpus = sysconf(_SC_NPROC_ONLN);

#elif HAVE_SYSCONF && defined(_SC_NPROCESSORS_ONLN)

    nb_cpus = sysconf(_SC_NPROCESSORS_ONLN);

#endif

    av_log(avctx, AV_LOG_DEBUG, "detected %d logical cores\n", nb_cpus);

    return FFMIN(nb_cpus, MAX_AUTO_THREADS);

}
