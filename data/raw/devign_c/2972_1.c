monitor_read_memory (bfd_vma memaddr, bfd_byte *myaddr, int length,

                     struct disassemble_info *info)

{

    CPUDebug *s = container_of(info, CPUDebug, info);



    if (monitor_disas_is_physical) {

        cpu_physical_memory_read(memaddr, myaddr, length);

    } else {

        cpu_memory_rw_debug(s->cpu, memaddr, myaddr, length, 0);

    }

    return 0;

}
