void parse_numa_opts(MachineClass *mc)

{

    int i;



    if (qemu_opts_foreach(qemu_find_opts("numa"), parse_numa, NULL, NULL)) {

        exit(1);

    }



    assert(max_numa_nodeid <= MAX_NODES);



    /* No support for sparse NUMA node IDs yet: */

    for (i = max_numa_nodeid - 1; i >= 0; i--) {

        /* Report large node IDs first, to make mistakes easier to spot */

        if (!numa_info[i].present) {

            error_report("numa: Node ID missing: %d", i);

            exit(1);

        }

    }



    /* This must be always true if all nodes are present: */

    assert(nb_numa_nodes == max_numa_nodeid);



    if (nb_numa_nodes > 0) {

        uint64_t numa_total;



        if (nb_numa_nodes > MAX_NODES) {

            nb_numa_nodes = MAX_NODES;

        }



        /* If no memory size is given for any node, assume the default case

         * and distribute the available memory equally across all nodes

         */

        for (i = 0; i < nb_numa_nodes; i++) {

            if (numa_info[i].node_mem != 0) {

                break;

            }

        }

        if (i == nb_numa_nodes) {

            uint64_t usedmem = 0;



            /* On Linux, each node's border has to be 8MB aligned,

             * the final node gets the rest.

             */

            for (i = 0; i < nb_numa_nodes - 1; i++) {

                numa_info[i].node_mem = (ram_size / nb_numa_nodes) &

                                        ~((1 << 23UL) - 1);

                usedmem += numa_info[i].node_mem;

            }

            numa_info[i].node_mem = ram_size - usedmem;

        }



        numa_total = 0;

        for (i = 0; i < nb_numa_nodes; i++) {

            numa_total += numa_info[i].node_mem;

        }

        if (numa_total != ram_size) {

            error_report("total memory for NUMA nodes (0x%" PRIx64 ")"

                         " should equal RAM size (0x" RAM_ADDR_FMT ")",

                         numa_total, ram_size);

            exit(1);

        }



        for (i = 0; i < nb_numa_nodes; i++) {

            QLIST_INIT(&numa_info[i].addr);

        }



        numa_set_mem_ranges();



        for (i = 0; i < nb_numa_nodes; i++) {

            if (!bitmap_empty(numa_info[i].node_cpu, MAX_CPUMASK_BITS)) {

                break;

            }

        }

        /* Historically VCPUs were assigned in round-robin order to NUMA

         * nodes. However it causes issues with guest not handling it nice

         * in case where cores/threads from a multicore CPU appear on

         * different nodes. So allow boards to override default distribution

         * rule grouping VCPUs by socket so that VCPUs from the same socket

         * would be on the same node.

         */

        if (i == nb_numa_nodes) {

            for (i = 0; i < max_cpus; i++) {

                unsigned node_id = i % nb_numa_nodes;

                if (mc->cpu_index_to_socket_id) {

                    node_id = mc->cpu_index_to_socket_id(i) % nb_numa_nodes;

                }



                set_bit(i, numa_info[node_id].node_cpu);

            }

        }



        validate_numa_cpus();

    } else {

        numa_set_mem_node_id(0, ram_size, 0);

    }

}
