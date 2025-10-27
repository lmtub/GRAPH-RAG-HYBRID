static void m68k_cpu_class_init(ObjectClass *c, void *data)
{
    M68kCPUClass *mcc = M68K_CPU_CLASS(c);
    CPUClass *cc = CPU_CLASS(c);
    DeviceClass *dc = DEVICE_CLASS(c);
    mcc->parent_realize = dc->realize;
    dc->realize = m68k_cpu_realizefn;
    mcc->parent_reset = cc->reset;
    cc->reset = m68k_cpu_reset;
    cc->class_by_name = m68k_cpu_class_by_name;
    cc->has_work = m68k_cpu_has_work;
    cc->do_interrupt = m68k_cpu_do_interrupt;
    cc->cpu_exec_interrupt = m68k_cpu_exec_interrupt;
    cc->dump_state = m68k_cpu_dump_state;
    cc->set_pc = m68k_cpu_set_pc;
    cc->gdb_read_register = m68k_cpu_gdb_read_register;
    cc->gdb_write_register = m68k_cpu_gdb_write_register;
#ifdef CONFIG_USER_ONLY
    cc->handle_mmu_fault = m68k_cpu_handle_mmu_fault;
#else
    cc->get_phys_page_debug = m68k_cpu_get_phys_page_debug;
#endif
    cc->cpu_exec_enter = m68k_cpu_exec_enter;
    cc->cpu_exec_exit = m68k_cpu_exec_exit;
    dc->vmsd = &vmstate_m68k_cpu;
    cc->gdb_num_core_regs = 18;
    cc->gdb_core_xml_file = "cf-core.xml";
}