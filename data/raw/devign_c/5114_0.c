int rpcit_service_call(S390CPU *cpu, uint8_t r1, uint8_t r2)

{

    CPUS390XState *env = &cpu->env;

    uint32_t fh;

    S390PCIBusDevice *pbdev;

    S390PCIIOMMU *iommu;

    hwaddr start, end;

    IOMMUTLBEntry entry;

    MemoryRegion *mr;



    cpu_synchronize_state(CPU(cpu));



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, 4);

        goto out;

    }



    if (r2 & 0x1) {

        program_interrupt(env, PGM_SPECIFICATION, 4);

        goto out;

    }



    fh = env->regs[r1] >> 32;

    start = env->regs[r2];

    end = start + env->regs[r2 + 1];



    pbdev = s390_pci_find_dev_by_fh(s390_get_phb(), fh);

    if (!pbdev) {

        DPRINTF("rpcit no pci dev\n");

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        goto out;

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

        s390_set_status_code(env, r1, ZPCI_MOD_ST_ERROR_RECOVER);

        return 0;

    default:

        break;

    }



    iommu = pbdev->iommu;

    if (!iommu->g_iota) {

        pbdev->state = ZPCI_FS_ERROR;

        setcc(cpu, ZPCI_PCI_LS_ERR);

        s390_set_status_code(env, r1, ZPCI_PCI_ST_INSUF_RES);

        s390_pci_generate_error_event(ERR_EVENT_INVALAS, pbdev->fh, pbdev->fid,

                                      start, 0);

        goto out;

    }



    if (end < iommu->pba || start > iommu->pal) {

        pbdev->state = ZPCI_FS_ERROR;

        setcc(cpu, ZPCI_PCI_LS_ERR);

        s390_set_status_code(env, r1, ZPCI_PCI_ST_INSUF_RES);

        s390_pci_generate_error_event(ERR_EVENT_OORANGE, pbdev->fh, pbdev->fid,

                                      start, 0);

        goto out;

    }



    mr = &iommu->iommu_mr;

    while (start < end) {

        entry = mr->iommu_ops->translate(mr, start, 0);



        if (!entry.translated_addr) {

            pbdev->state = ZPCI_FS_ERROR;

            setcc(cpu, ZPCI_PCI_LS_ERR);

            s390_set_status_code(env, r1, ZPCI_PCI_ST_INSUF_RES);

            s390_pci_generate_error_event(ERR_EVENT_SERR, pbdev->fh, pbdev->fid,

                                          start, ERR_EVENT_Q_BIT);

            goto out;

        }



        memory_region_notify_iommu(mr, entry);

        start += entry.addr_mask + 1;

    }



    setcc(cpu, ZPCI_PCI_LS_OK);

out:

    return 0;

}
