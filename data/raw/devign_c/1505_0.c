static void spapr_cpu_core_realize(DeviceState *dev, Error **errp)

{

    /* We don't use SPAPR_MACHINE() in order to exit gracefully if the user

     * tries to add a sPAPR CPU core to a non-pseries machine.

     */

    sPAPRMachineState *spapr =

        (sPAPRMachineState *) object_dynamic_cast(qdev_get_machine(),

                                                  TYPE_SPAPR_MACHINE);

    sPAPRCPUCore *sc = SPAPR_CPU_CORE(OBJECT(dev));

    sPAPRCPUCoreClass *scc = SPAPR_CPU_CORE_GET_CLASS(OBJECT(dev));

    CPUCore *cc = CPU_CORE(OBJECT(dev));

    size_t size;

    Error *local_err = NULL;

    void *obj;

    int i, j;



    if (!spapr) {

        error_setg(errp, TYPE_SPAPR_CPU_CORE " needs a pseries machine");

        return;

    }



    size = object_type_get_instance_size(scc->cpu_type);

    sc->threads = g_malloc0(size * cc->nr_threads);

    for (i = 0; i < cc->nr_threads; i++) {

        char id[32];

        CPUState *cs;

        PowerPCCPU *cpu;



        obj = sc->threads + i * size;



        object_initialize(obj, size, scc->cpu_type);

        cs = CPU(obj);

        cpu = POWERPC_CPU(cs);

        cs->cpu_index = cc->core_id + i;

        cpu->vcpu_id = (cc->core_id * spapr->vsmt / smp_threads) + i;

        if (kvm_enabled() && !kvm_vcpu_id_is_valid(cpu->vcpu_id)) {

            error_setg(&local_err, "Can't create CPU with id %d in KVM",

                       cpu->vcpu_id);

            error_append_hint(&local_err, "Adjust the number of cpus to %d "

                              "or try to raise the number of threads per core\n",

                              cpu->vcpu_id * smp_threads / spapr->vsmt);

            goto err;

        }





        /* Set NUMA node for the threads belonged to core  */

        cpu->node_id = sc->node_id;



        snprintf(id, sizeof(id), "thread[%d]", i);

        object_property_add_child(OBJECT(sc), id, obj, &local_err);

        if (local_err) {

            goto err;

        }

        object_unref(obj);

    }



    for (j = 0; j < cc->nr_threads; j++) {

        obj = sc->threads + j * size;



        spapr_cpu_core_realize_child(obj, spapr, &local_err);

        if (local_err) {

            goto err;

        }

    }

    return;



err:

    while (--i >= 0) {

        obj = sc->threads + i * size;

        object_unparent(obj);

    }

    g_free(sc->threads);

    error_propagate(errp, local_err);

}
