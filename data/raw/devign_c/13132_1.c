int pcistb_service_call(S390CPU *cpu, uint8_t r1, uint8_t r3, uint64_t gaddr,

                        uint8_t ar)

{

    CPUS390XState *env = &cpu->env;

    S390PCIBusDevice *pbdev;

    MemoryRegion *mr;

    int i;

    uint32_t fh;

    uint8_t pcias;

    uint8_t len;

    uint8_t buffer[128];



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, 6);

        return 0;

    }



    fh = env->regs[r1] >> 32;

    pcias = (env->regs[r1] >> 16) & 0xf;

    len = env->regs[r1] & 0xff;



    if (pcias > 5) {

        DPRINTF("pcistb invalid space\n");

        setcc(cpu, ZPCI_PCI_LS_ERR);

        s390_set_status_code(env, r1, ZPCI_PCI_ST_INVAL_AS);

        return 0;

    }



    switch (len) {

    case 16:

    case 32:

    case 64:

    case 128:

        break;

    default:

        program_interrupt(env, PGM_SPECIFICATION, 6);

        return 0;

    }



    pbdev = s390_pci_find_dev_by_fh(fh);

    if (!pbdev) {

        DPRINTF("pcistb no pci dev fh 0x%x\n", fh);

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    }



    switch (pbdev->state) {

    case ZPCI_FS_RESERVED:

    case ZPCI_FS_STANDBY:

    case ZPCI_FS_DISABLED:

    case ZPCI_FS_PERMANENT_ERROR:

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    case ZPCI_FS_ERROR:

        setcc(cpu, ZPCI_PCI_LS_ERR);

        s390_set_status_code(env, r1, ZPCI_PCI_ST_BLOCKED);

        return 0;

    default:

        break;

    }



    mr = pbdev->pdev->io_regions[pcias].memory;

    if (!memory_region_access_valid(mr, env->regs[r3], len, true)) {

        program_interrupt(env, PGM_ADDRESSING, 6);

        return 0;

    }



    if (s390_cpu_virt_mem_read(cpu, gaddr, ar, buffer, len)) {

        return 0;

    }



    for (i = 0; i < len / 8; i++) {

        memory_region_dispatch_write(mr, env->regs[r3] + i * 8,

                                     ldq_p(buffer + i * 8), 8,

                                     MEMTXATTRS_UNSPECIFIED);

    }



    setcc(cpu, ZPCI_PCI_LS_OK);

    return 0;

}
