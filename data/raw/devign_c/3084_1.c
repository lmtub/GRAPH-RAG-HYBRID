static void xtensa_cpu_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    CPUClass *cc = CPU_CLASS(oc);

    XtensaCPUClass *xcc = XTENSA_CPU_CLASS(cc);



    xcc->parent_realize = dc->realize;

    dc->realize = xtensa_cpu_realizefn;



    xcc->parent_reset = cc->reset;

    cc->reset = xtensa_cpu_reset;



    cc->class_by_name = xtensa_cpu_class_by_name;

    cc->has_work = xtensa_cpu_has_work;

    cc->do_interrupt = xtensa_cpu_do_interrupt;

    cc->cpu_exec_interrupt = xtensa_cpu_exec_interrupt;

    cc->dump_state = xtensa_cpu_dump_state;

    cc->set_pc = xtensa_cpu_set_pc;

    cc->gdb_read_register = xtensa_cpu_gdb_read_register;

    cc->gdb_write_register = xtensa_cpu_gdb_write_register;

    cc->gdb_stop_before_watchpoint = true;

#ifndef CONFIG_USER_ONLY

    cc->do_unaligned_access = xtensa_cpu_do_unaligned_access;

    cc->get_phys_page_debug = xtensa_cpu_get_phys_page_debug;

    cc->do_unassigned_access = xtensa_cpu_do_unassigned_access;

#endif

    cc->debug_excp_handler = xtensa_breakpoint_handler;

    dc->vmsd = &vmstate_xtensa_cpu;



    /*

     * Reason: xtensa_cpu_initfn() calls cpu_exec_init(), which saves

     * the object in cpus -> dangling pointer after final

     * object_unref().

     */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
