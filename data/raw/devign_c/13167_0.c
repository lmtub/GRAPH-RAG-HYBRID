static void spapr_core_pre_plug(HotplugHandler *hotplug_dev, DeviceState *dev,

                                Error **errp)

{

    MachineState *machine = MACHINE(OBJECT(hotplug_dev));

    MachineClass *mc = MACHINE_GET_CLASS(hotplug_dev);

    Error *local_err = NULL;

    CPUCore *cc = CPU_CORE(dev);

    sPAPRCPUCore *sc = SPAPR_CPU_CORE(dev);

    char *base_core_type = spapr_get_cpu_core_type(machine->cpu_model);

    const char *type = object_get_typename(OBJECT(dev));

    CPUArchId *core_slot;

    int node_id;

    int index;



    if (dev->hotplugged && !mc->has_hotpluggable_cpus) {

        error_setg(&local_err, "CPU hotplug not supported for this machine");

        goto out;

    }



    if (strcmp(base_core_type, type)) {

        error_setg(&local_err, "CPU core type should be %s", base_core_type);

        goto out;

    }



    if (cc->core_id % smp_threads) {

        error_setg(&local_err, "invalid core id %d", cc->core_id);

        goto out;

    }



    if (cc->nr_threads != smp_threads) {

        error_setg(errp, "invalid nr-threads %d, must be %d",

                   cc->nr_threads, smp_threads);

        return;

    }



    core_slot = spapr_find_cpu_slot(MACHINE(hotplug_dev), cc->core_id, &index);

    if (!core_slot) {

        error_setg(&local_err, "core id %d out of range", cc->core_id);

        goto out;

    }



    if (core_slot->cpu) {

        error_setg(&local_err, "core %d already populated", cc->core_id);

        goto out;

    }



    node_id = numa_get_node_for_cpu(cc->core_id);

    if (node_id == nb_numa_nodes) {

        /* by default CPUState::numa_node was 0 if it's not set via CLI

         * keep it this way for now but in future we probably should

         * refuse to start up with incomplete numa mapping */

        node_id = 0;

    }

    if (sc->node_id == CPU_UNSET_NUMA_NODE_ID) {

        sc->node_id = node_id;

    } else if (sc->node_id != node_id) {

        error_setg(&local_err, "node-id %d must match numa node specified"

            "with -numa option for cpu-index %d", sc->node_id, cc->core_id);

        goto out;

    }



out:

    g_free(base_core_type);

    error_propagate(errp, local_err);

}
