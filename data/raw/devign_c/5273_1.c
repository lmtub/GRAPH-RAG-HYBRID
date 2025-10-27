static void sparc_cpu_class_init(ObjectClass *oc, void *data)

{

    SPARCCPUClass *scc = SPARC_CPU_CLASS(oc);

    CPUClass *cc = CPU_CLASS(oc);

    DeviceClass *dc = DEVICE_CLASS(oc);



    scc->parent_realize = dc->realize;

    dc->realize = sparc_cpu_realizefn;



    scc->parent_reset = cc->reset;

    cc->reset = sparc_cpu_reset;



    cc->has_work = sparc_cpu_has_work;

    cc->do_interrupt = sparc_cpu_do_interrupt;

    cc->cpu_exec_interrupt = sparc_cpu_exec_interrupt;

    cc->dump_state = sparc_cpu_dump_state;

#if !defined(TARGET_SPARC64) && !defined(CONFIG_USER_ONLY)

    cc->memory_rw_debug = sparc_cpu_memory_rw_debug;

#endif

    cc->set_pc = sparc_cpu_set_pc;

    cc->synchronize_from_tb = sparc_cpu_synchronize_from_tb;

    cc->gdb_read_register = sparc_cpu_gdb_read_register;

    cc->gdb_write_register = sparc_cpu_gdb_write_register;

#ifdef CONFIG_USER_ONLY

    cc->handle_mmu_fault = sparc_cpu_handle_mmu_fault;

#else

    cc->do_unassigned_access = sparc_cpu_unassigned_access;

    cc->do_unaligned_access = sparc_cpu_do_unaligned_access;

    cc->get_phys_page_debug = sparc_cpu_get_phys_page_debug;

    cc->vmsd = &vmstate_sparc_cpu;

#endif

    cc->disas_set_info = cpu_sparc_disas_set_info;



#if defined(TARGET_SPARC64) && !defined(TARGET_ABI32)

    cc->gdb_num_core_regs = 86;

#else

    cc->gdb_num_core_regs = 72;

#endif



    /*

     * Reason: sparc_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
