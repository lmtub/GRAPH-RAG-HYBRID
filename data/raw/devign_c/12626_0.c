static void validate_numa_cpus(void)

{

    int i;

    unsigned long *seen_cpus = bitmap_new(max_cpus);



    for (i = 0; i < nb_numa_nodes; i++) {

        if (bitmap_intersects(seen_cpus, numa_info[i].node_cpu, max_cpus)) {

            bitmap_and(seen_cpus, seen_cpus,

                       numa_info[i].node_cpu, max_cpus);

            error_report("CPU(s) present in multiple NUMA nodes: %s",

                         enumerate_cpus(seen_cpus, max_cpus));

            g_free(seen_cpus);

            exit(EXIT_FAILURE);

        }

        bitmap_or(seen_cpus, seen_cpus,

                  numa_info[i].node_cpu, max_cpus);

    }



    if (!bitmap_full(seen_cpus, max_cpus)) {

        char *msg;

        bitmap_complement(seen_cpus, seen_cpus, max_cpus);

        msg = enumerate_cpus(seen_cpus, max_cpus);

        error_report("warning: CPU(s) not present in any NUMA nodes: %s", msg);

        error_report("warning: All CPU(s) up to maxcpus should be described "

                     "in NUMA config");

        g_free(msg);

    }

    g_free(seen_cpus);

}
