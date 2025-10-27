static void smp_parse(const char *optarg)

{

    int smp, sockets = 0, threads = 0, cores = 0;

    char *endptr;

    char option[128];



    smp = strtoul(optarg, &endptr, 10);

    if (endptr != optarg) {

        if (*endptr == ',') {

            endptr++;

        }

    }

    if (get_param_value(option, 128, "sockets", endptr) != 0)

        sockets = strtoull(option, NULL, 10);

    if (get_param_value(option, 128, "cores", endptr) != 0)

        cores = strtoull(option, NULL, 10);

    if (get_param_value(option, 128, "threads", endptr) != 0)

        threads = strtoull(option, NULL, 10);

    if (get_param_value(option, 128, "maxcpus", endptr) != 0)

        max_cpus = strtoull(option, NULL, 10);



    /* compute missing values, prefer sockets over cores over threads */

    if (smp == 0 || sockets == 0) {

        sockets = sockets > 0 ? sockets : 1;

        cores = cores > 0 ? cores : 1;

        threads = threads > 0 ? threads : 1;

        if (smp == 0) {

            smp = cores * threads * sockets;

        }

    } else {

        if (cores == 0) {

            threads = threads > 0 ? threads : 1;

            cores = smp / (sockets * threads);

        } else {

            threads = smp / (cores * sockets);

        }

    }

    smp_cpus = smp;

    smp_cores = cores > 0 ? cores : 1;

    smp_threads = threads > 0 ? threads : 1;

    if (max_cpus == 0)

        max_cpus = smp_cpus;

}
