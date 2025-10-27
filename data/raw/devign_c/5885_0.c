static void numa_add(const char *optarg)

{

    char option[128];

    char *endptr;

    unsigned long long value, endvalue;

    int nodenr;



    optarg = get_opt_name(option, 128, optarg, ',') + 1;

    if (!strcmp(option, "node")) {

        if (get_param_value(option, 128, "nodeid", optarg) == 0) {

            nodenr = nb_numa_nodes;

        } else {

            nodenr = strtoull(option, NULL, 10);

        }



        if (get_param_value(option, 128, "mem", optarg) == 0) {

            node_mem[nodenr] = 0;

        } else {

            int64_t sval;

            sval = strtosz(option, &endptr);

            if (sval < 0 || *endptr) {

                fprintf(stderr, "qemu: invalid numa mem size: %s\n", optarg);

                exit(1);

            }

            node_mem[nodenr] = sval;

        }

        if (get_param_value(option, 128, "cpus", optarg) == 0) {

            node_cpumask[nodenr] = 0;

        } else {

            value = strtoull(option, &endptr, 10);

            if (value >= 64) {

                value = 63;

                fprintf(stderr, "only 64 CPUs in NUMA mode supported.\n");

            } else {

                if (*endptr == '-') {

                    endvalue = strtoull(endptr+1, &endptr, 10);

                    if (endvalue >= 63) {

                        endvalue = 62;

                        fprintf(stderr,

                            "only 63 CPUs in NUMA mode supported.\n");

                    }

                    value = (2ULL << endvalue) - (1ULL << value);

                } else {

                    value = 1ULL << value;

                }

            }

            node_cpumask[nodenr] = value;

        }

        nb_numa_nodes++;

    }

    return;

}
