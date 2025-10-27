static void aarch64_cpu_class_init(ObjectClass *oc, void *data)

{

    CPUClass *cc = CPU_CLASS(oc);




    cc->dump_state = aarch64_cpu_dump_state;

    cc->set_pc = aarch64_cpu_set_pc;

    cc->gdb_read_register = aarch64_cpu_gdb_read_register;

    cc->gdb_write_register = aarch64_cpu_gdb_write_register;

    cc->gdb_num_core_regs = 34;

    cc->gdb_core_xml_file = "aarch64-core.xml";

}