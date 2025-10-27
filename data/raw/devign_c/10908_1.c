static void spapr_core_plug(HotplugHandler *hotplug_dev, DeviceState *dev,

                            Error **errp)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(OBJECT(hotplug_dev));

    MachineClass *mc = MACHINE_GET_CLASS(spapr);

    sPAPRMachineClass *smc = SPAPR_MACHINE_CLASS(mc);

    sPAPRCPUCore *core = SPAPR_CPU_CORE(OBJECT(dev));

    CPUCore *cc = CPU_CORE(dev);

    CPUState *cs = CPU(core->threads);

    sPAPRDRConnector *drc;

    Error *local_err = NULL;

    void *fdt = NULL;

    int fdt_offset = 0;

    int smt = kvmppc_smt_threads();

    CPUArchId *core_slot;

    int index;

    bool hotplugged = spapr_drc_hotplugged(dev);



    core_slot = spapr_find_cpu_slot(MACHINE(hotplug_dev), cc->core_id, &index);

    if (!core_slot) {

        error_setg(errp, "Unable to find CPU core with core-id: %d",

                   cc->core_id);

        return;

    }

    drc = spapr_drc_by_id(TYPE_SPAPR_DRC_CPU, index * smt);



    g_assert(drc || !mc->has_hotpluggable_cpus);



    fdt = spapr_populate_hotplug_cpu_dt(cs, &fdt_offset, spapr);



    if (drc) {

        spapr_drc_attach(drc, dev, fdt, fdt_offset, &local_err);

        if (local_err) {

            g_free(fdt);

            error_propagate(errp, local_err);

            return;

        }



        if (hotplugged) {

            /*

             * Send hotplug notification interrupt to the guest only

             * in case of hotplugged CPUs.

             */

            spapr_hotplug_req_add_by_index(drc);

        } else {

            spapr_drc_reset(drc);

        }

    }



    core_slot->cpu = OBJECT(dev);



    if (smc->pre_2_10_has_unused_icps) {

        sPAPRCPUCoreClass *scc = SPAPR_CPU_CORE_GET_CLASS(OBJECT(cc));

        const char *typename = object_class_get_name(scc->cpu_class);

        size_t size = object_type_get_instance_size(typename);

        int i;



        for (i = 0; i < cc->nr_threads; i++) {

            sPAPRCPUCore *sc = SPAPR_CPU_CORE(dev);

            void *obj = sc->threads + i * size;



            cs = CPU(obj);

            pre_2_10_vmstate_unregister_dummy_icp(cs->cpu_index);

        }

    }

}
