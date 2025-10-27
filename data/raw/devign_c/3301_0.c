int cpu_get_dump_info(ArchDumpInfo *info,

                      const struct GuestPhysBlockList *guest_phys_blocks)

{

    PowerPCCPU *cpu = POWERPC_CPU(first_cpu);

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cpu);



    info->d_machine = PPC_ELF_MACHINE;

    info->d_class = ELFCLASS;



    if ((*pcc->interrupts_big_endian)(cpu)) {

        info->d_endian = ELFDATA2MSB;

    } else {

        info->d_endian = ELFDATA2LSB;

    }

    /* 64KB is the max page size for pseries kernel */

    if (strncmp(object_get_typename(qdev_get_machine()),

                "pseries-", 8) == 0) {

        info->page_size = (1U << 16);

    }



    return 0;

}
