static void s390_cpu_class_init(ObjectClass *oc, void *data)

{

    S390CPUClass *scc = S390_CPU_CLASS(oc);

    CPUClass *cc = CPU_CLASS(scc);

    DeviceClass *dc = DEVICE_CLASS(oc);



    scc->parent_realize = dc->realize;

    dc->realize = s390_cpu_realizefn;



    scc->parent_reset = cc->reset;

    cc->reset = s390_cpu_reset;



    cc->do_interrupt = s390_cpu_do_interrupt;

    cc->dump_state = s390_cpu_dump_state;

    cc->set_pc = s390_cpu_set_pc;

    cc->gdb_read_register = s390_cpu_gdb_read_register;

    cc->gdb_write_register = s390_cpu_gdb_write_register;

#ifndef CONFIG_USER_ONLY

    cc->get_phys_page_debug = s390_cpu_get_phys_page_debug;



#endif

    dc->vmsd = &vmstate_s390_cpu;

    cc->gdb_num_core_regs = S390_NUM_REGS;

}