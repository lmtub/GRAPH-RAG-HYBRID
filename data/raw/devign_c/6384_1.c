static HotpluggableCPUList *spapr_query_hotpluggable_cpus(MachineState *machine)

{

    int i;

    HotpluggableCPUList *head = NULL;

    sPAPRMachineState *spapr = SPAPR_MACHINE(machine);

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(machine);

    int spapr_max_cores = max_cpus / smp_threads;



    g_assert(smc->dr_cpu_enabled);



    for (i = 0; i < spapr_max_cores; i++) {

        HotpluggableCPUList *list_item = g_new0(typeof(*list_item), 1);

        HotpluggableCPU *cpu_item = g_new0(typeof(*cpu_item), 1);

        CpuInstanceProperties *cpu_props = g_new0(typeof(*cpu_props), 1);



        cpu_item->type = spapr_get_cpu_core_type(machine->cpu_model);

        cpu_item->vcpus_count = smp_threads;

        cpu_props->has_core_id = true;

        cpu_props->core_id = i * smp_threads;

        /* TODO: add 'has_node/node' here to describe

           to which node core belongs */



        cpu_item->props = cpu_props;

        if (spapr->cores[i]) {

            cpu_item->has_qom_path = true;

            cpu_item->qom_path = object_get_canonical_path(spapr->cores[i]);

        }

        list_item->value = cpu_item;

        list_item->next = head;

        head = list_item;

    }

    return head;

}
