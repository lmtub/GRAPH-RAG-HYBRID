void spapr_core_release(DeviceState *dev)

{

    MachineState *ms = MACHINE(qdev_get_hotplug_handler(dev));

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(ms);

    CPUCore *cc = CPU_CORE(dev);

    CPUArchId *core_slot = spapr_find_cpu_slot(ms, cc->core_id, NULL);



    if (smc->pre_2_10_has_unused_icps) {

        sPAPRCPUCore *sc = SPAPR_CPU_CORE(OBJECT(dev));

        sPAPRCPUCoreClass *scc = SPAPR_CPU_CORE_GET_CLASS(OBJECT(cc));

        size_t size = object_type_get_instance_size(scc->cpu_type);

        int i;



        for (i = 0; i < cc->nr_threads; i++) {

            CPUState *cs = CPU(sc->threads + i * size);



            pre_2_10_vmstate_register_dummy_icp(cs->cpu_index);

        }

    }



    assert(core_slot);

    core_slot->cpu = NULL;

    object_unparent(OBJECT(dev));

}
