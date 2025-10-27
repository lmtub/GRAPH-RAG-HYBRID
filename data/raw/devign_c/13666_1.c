static void moxie_cpu_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    CPUClass *cc = CPU_CLASS(oc);

    MoxieCPUClass *mcc = MOXIE_CPU_CLASS(oc);



    mcc->parent_realize = dc->realize;

    dc->realize = moxie_cpu_realizefn;



    mcc->parent_reset = cc->reset;

    cc->reset = moxie_cpu_reset;



    cc->class_by_name = moxie_cpu_class_by_name;



    cc->has_work = moxie_cpu_has_work;

    cc->do_interrupt = moxie_cpu_do_interrupt;

    cc->dump_state = moxie_cpu_dump_state;

    cc->set_pc = moxie_cpu_set_pc;

#ifdef CONFIG_USER_ONLY

    cc->handle_mmu_fault = moxie_cpu_handle_mmu_fault;

#else

    cc->get_phys_page_debug = moxie_cpu_get_phys_page_debug;

    cc->vmsd = &vmstate_moxie_cpu;

#endif

    cc->disas_set_info = moxie_cpu_disas_set_info;



    /*

     * Reason: moxie_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
