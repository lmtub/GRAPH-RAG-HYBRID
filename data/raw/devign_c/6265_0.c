static void machine_numa_validate(MachineState *machine)

{

    int i;

    GString *s = g_string_new(NULL);

    MachineClass *mc = MACHINE_GET_CLASS(machine);

    const CPUArchIdList *possible_cpus = mc->possible_cpu_arch_ids(machine);



    assert(nb_numa_nodes);

    for (i = 0; i < possible_cpus->len; i++) {

        const CPUArchId *cpu_slot = &possible_cpus->cpus[i];



        /* at this point numa mappings are initilized by CLI options

         * or with default mappings so it's sufficient to list

         * all not yet mapped CPUs here */

        /* TODO: make it hard error in future */

        if (!cpu_slot->props.has_node_id) {

            char *cpu_str = cpu_slot_to_string(cpu_slot);

            g_string_append_printf(s, "%sCPU %d [%s]", s->len ? ", " : "", i,

                                   cpu_str);

            g_free(cpu_str);

        }

    }

    if (s->len) {

        error_report("warning: CPU(s) not present in any NUMA nodes: %s",

                     s->str);

        error_report("warning: All CPU(s) up to maxcpus should be described "

                     "in NUMA config, ability to start up with partial NUMA "

                     "mappings is obsoleted and will be removed in future");

    }

    g_string_free(s, true);

}
