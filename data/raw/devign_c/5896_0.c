static void machine_numa_finish_init(MachineState *machine)

{

    int i;

    bool default_mapping;

    GString *s = g_string_new(NULL);

    MachineClass *mc = MACHINE_GET_CLASS(machine);

    const CPUArchIdList *possible_cpus = mc->possible_cpu_arch_ids(machine);



    assert(nb_numa_nodes);

    for (i = 0; i < possible_cpus->len; i++) {

        if (possible_cpus->cpus[i].props.has_node_id) {

            break;

        }

    }

    default_mapping = (i == possible_cpus->len);



    for (i = 0; i < possible_cpus->len; i++) {

        const CPUArchId *cpu_slot = &possible_cpus->cpus[i];



        if (!cpu_slot->props.has_node_id) {

            if (default_mapping) {

                /* fetch default mapping from board and enable it */

                CpuInstanceProperties props = cpu_slot->props;

                props.has_node_id = true;

                machine_set_cpu_numa_node(machine, &props, &error_fatal);

            } else {

                /* record slots with not set mapping,

                 * TODO: make it hard error in future */

                char *cpu_str = cpu_slot_to_string(cpu_slot);

                g_string_append_printf(s, "%sCPU %d [%s]",

                                       s->len ? ", " : "", i, cpu_str);

                g_free(cpu_str);

            }

        }

    }

    if (s->len && !qtest_enabled()) {

        error_report("warning: CPU(s) not present in any NUMA nodes: %s",

                     s->str);

        error_report("warning: All CPU(s) up to maxcpus should be described "

                     "in NUMA config, ability to start up with partial NUMA "

                     "mappings is obsoleted and will be removed in future");

    }

    g_string_free(s, true);

}
