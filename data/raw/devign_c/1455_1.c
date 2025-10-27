int stpcifc_service_call(S390CPU *cpu, uint8_t r1, uint64_t fiba, uint8_t ar)

{

    CPUS390XState *env = &cpu->env;

    uint32_t fh;

    ZpciFib fib;

    S390PCIBusDevice *pbdev;

    uint32_t data;

    uint64_t cc = ZPCI_PCI_LS_OK;



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, 6);

        return 0;

    }



    fh = env->regs[r1] >> 32;



    if (fiba & 0x7) {

        program_interrupt(env, PGM_SPECIFICATION, 6);

        return 0;

    }



    pbdev = s390_pci_find_dev_by_fh(fh);

    if (!pbdev) {

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    }



    memset(&fib, 0, sizeof(fib));



    switch (pbdev->state) {

    case ZPCI_FS_RESERVED:

    case ZPCI_FS_STANDBY:

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    case ZPCI_FS_DISABLED:

        if (fh & FH_MASK_ENABLE) {

            setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

            return 0;

        }

        goto out;

    /* BLOCKED bit is set to one coincident with the setting of ERROR bit.

     * FH Enabled bit is set to one in states of ENABLED, BLOCKED or ERROR. */

    case ZPCI_FS_ERROR:

        fib.fc |= 0x20;

    case ZPCI_FS_BLOCKED:

        fib.fc |= 0x40;

    case ZPCI_FS_ENABLED:

        fib.fc |= 0x80;

        if (pbdev->iommu_enabled) {

            fib.fc |= 0x10;

        }

        if (!(fh & FH_MASK_ENABLE)) {

            env->regs[r1] |= 1ULL << 63;

        }

        break;

    case ZPCI_FS_PERMANENT_ERROR:

        setcc(cpu, ZPCI_PCI_LS_ERR);

        s390_set_status_code(env, r1, ZPCI_STPCIFC_ST_PERM_ERROR);

        return 0;

    }



    stq_p(&fib.pba, pbdev->pba);

    stq_p(&fib.pal, pbdev->pal);

    stq_p(&fib.iota, pbdev->g_iota);

    stq_p(&fib.aibv, pbdev->routes.adapter.ind_addr);

    stq_p(&fib.aisb, pbdev->routes.adapter.summary_addr);

    stq_p(&fib.fmb_addr, pbdev->fmb_addr);



    data = ((uint32_t)pbdev->isc << 28) | ((uint32_t)pbdev->noi << 16) |

           ((uint32_t)pbdev->routes.adapter.ind_offset << 8) |

           ((uint32_t)pbdev->sum << 7) | pbdev->routes.adapter.summary_offset;

    stl_p(&fib.data, data);



out:

    if (s390_cpu_virt_mem_write(cpu, fiba, ar, (uint8_t *)&fib, sizeof(fib))) {

        return 0;

    }



    setcc(cpu, cc);

    return 0;

}
