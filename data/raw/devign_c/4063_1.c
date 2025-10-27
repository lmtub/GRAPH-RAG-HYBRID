static void superh_cpu_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    CPUClass *cc = CPU_CLASS(oc);

    SuperHCPUClass *scc = SUPERH_CPU_CLASS(oc);



    scc->parent_realize = dc->realize;

    dc->realize = superh_cpu_realizefn;



    scc->parent_reset = cc->reset;

    cc->reset = superh_cpu_reset;



    cc->class_by_name = superh_cpu_class_by_name;

    cc->has_work = superh_cpu_has_work;

    cc->do_interrupt = superh_cpu_do_interrupt;

    cc->cpu_exec_interrupt = superh_cpu_exec_interrupt;

    cc->dump_state = superh_cpu_dump_state;

    cc->set_pc = superh_cpu_set_pc;

    cc->synchronize_from_tb = superh_cpu_synchronize_from_tb;

    cc->gdb_read_register = superh_cpu_gdb_read_register;

    cc->gdb_write_register = superh_cpu_gdb_write_register;

#ifdef CONFIG_USER_ONLY

    cc->handle_mmu_fault = superh_cpu_handle_mmu_fault;

#else

    cc->get_phys_page_debug = superh_cpu_get_phys_page_debug;

#endif

    cc->disas_set_info = superh_cpu_disas_set_info;



    cc->gdb_num_core_regs = 59;



    dc->vmsd = &vmstate_sh_cpu;



    /*

     * Reason: superh_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
