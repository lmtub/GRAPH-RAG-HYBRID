static void cpu_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    CPUClass *k = CPU_CLASS(klass);



    k->class_by_name = cpu_common_class_by_name;

    k->reset = cpu_common_reset;

    k->get_arch_id = cpu_common_get_arch_id;

    k->get_paging_enabled = cpu_common_get_paging_enabled;

    k->get_memory_mapping = cpu_common_get_memory_mapping;

    k->write_elf32_qemunote = cpu_common_write_elf32_qemunote;

    k->write_elf32_note = cpu_common_write_elf32_note;

    k->write_elf64_qemunote = cpu_common_write_elf64_qemunote;

    k->write_elf64_note = cpu_common_write_elf64_note;

    k->gdb_read_register = cpu_common_gdb_read_register;

    k->gdb_write_register = cpu_common_gdb_write_register;

    dc->realize = cpu_common_realizefn;

    dc->no_user = 1;

}
