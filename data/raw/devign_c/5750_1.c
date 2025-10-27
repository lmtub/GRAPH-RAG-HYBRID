static void smp_parse(QemuOpts *opts)

{

    if (opts) {



        unsigned cpus    = qemu_opt_get_number(opts, "cpus", 0);

        unsigned sockets = qemu_opt_get_number(opts, "sockets", 0);

        unsigned cores   = qemu_opt_get_number(opts, "cores", 0);

        unsigned threads = qemu_opt_get_number(opts, "threads", 0);



        /* compute missing values, prefer sockets over cores over threads */

        if (cpus == 0 || sockets == 0) {

            sockets = sockets > 0 ? sockets : 1;

            cores = cores > 0 ? cores : 1;

            threads = threads > 0 ? threads : 1;

            if (cpus == 0) {

                cpus = cores * threads * sockets;

            }

        } else if (cores == 0) {

            threads = threads > 0 ? threads : 1;

            cores = cpus / (sockets * threads);

        } else if (threads == 0) {

            threads = cpus / (cores * sockets);

        } else if (sockets * cores * threads < cpus) {

            fprintf(stderr, "cpu topology: error: "

                    "sockets (%u) * cores (%u) * threads (%u) < "

                    "smp_cpus (%u)\n",

                    sockets, cores, threads, cpus);

            exit(1);

        }



        max_cpus = qemu_opt_get_number(opts, "maxcpus", 0);



        smp_cpus = cpus;

        smp_cores = cores > 0 ? cores : 1;

        smp_threads = threads > 0 ? threads : 1;



    }



    if (max_cpus == 0) {

        max_cpus = smp_cpus;

    }



    if (max_cpus > MAX_CPUMASK_BITS) {

        fprintf(stderr, "Unsupported number of maxcpus\n");

        exit(1);

    }

    if (max_cpus < smp_cpus) {

        fprintf(stderr, "maxcpus must be equal to or greater than smp\n");

        exit(1);

    }



}
