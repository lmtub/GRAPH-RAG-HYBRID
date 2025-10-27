static void mips_cpu_class_init(ObjectClass *c, void *data)

{

    MIPSCPUClass *mcc = MIPS_CPU_CLASS(c);

    CPUClass *cc = CPU_CLASS(c);

    DeviceClass *dc = DEVICE_CLASS(c);



    mcc->parent_realize = dc->realize;

    dc->realize = mips_cpu_realizefn;



    mcc->parent_reset = cc->reset;

    cc->reset = mips_cpu_reset;



    cc->has_work = mips_cpu_has_work;

    cc->do_interrupt = mips_cpu_do_interrupt;

    cc->cpu_exec_interrupt = mips_cpu_exec_interrupt;

    cc->dump_state = mips_cpu_dump_state;

    cc->set_pc = mips_cpu_set_pc;

    cc->synchronize_from_tb = mips_cpu_synchronize_from_tb;

    cc->gdb_read_register = mips_cpu_gdb_read_register;

    cc->gdb_write_register = mips_cpu_gdb_write_register;

#ifdef CONFIG_USER_ONLY

    cc->handle_mmu_fault = mips_cpu_handle_mmu_fault;

#else

    cc->do_unassigned_access = mips_cpu_unassigned_access;

    cc->do_unaligned_access = mips_cpu_do_unaligned_access;

    cc->get_phys_page_debug = mips_cpu_get_phys_page_debug;

    cc->vmsd = &vmstate_mips_cpu;

#endif

    cc->disas_set_info = mips_cpu_disas_set_info;



    cc->gdb_num_core_regs = 73;

    cc->gdb_stop_before_watchpoint = true;



    /*

     * Reason: mips_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
