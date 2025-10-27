static void rtas_ibm_change_msi(PowerPCCPU *cpu, sPAPREnvironment *spapr,

                                uint32_t token, uint32_t nargs,

                                target_ulong args, uint32_t nret,

                                target_ulong rets)

{

    uint32_t config_addr = rtas_ld(args, 0);

    uint64_t buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    unsigned int func = rtas_ld(args, 3);

    unsigned int req_num = rtas_ld(args, 4); /* 0 == remove all */

    unsigned int seq_num = rtas_ld(args, 5);

    unsigned int ret_intr_type;

    int ndev, irq, max_irqs = 0;

    sPAPRPHBState *phb = NULL;

    PCIDevice *pdev = NULL;



    switch (func) {

    case RTAS_CHANGE_MSI_FN:

    case RTAS_CHANGE_FN:

        ret_intr_type = RTAS_TYPE_MSI;

        break;

    case RTAS_CHANGE_MSIX_FN:

        ret_intr_type = RTAS_TYPE_MSIX;

        break;

    default:

        error_report("rtas_ibm_change_msi(%u) is not implemented", func);

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    /* Fins sPAPRPHBState */

    phb = find_phb(spapr, buid);

    if (phb) {

        pdev = find_dev(spapr, buid, config_addr);

    }

    if (!phb || !pdev) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    /* Releasing MSIs */

    if (!req_num) {

        ndev = spapr_msicfg_find(phb, config_addr, false);

        if (ndev < 0) {

            trace_spapr_pci_msi("MSI has not been enabled", -1, config_addr);

            rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

            return;

        }

        trace_spapr_pci_msi("Released MSIs", ndev, config_addr);

        rtas_st(rets, 0, RTAS_OUT_SUCCESS);

        rtas_st(rets, 1, 0);

        return;

    }



    /* Enabling MSI */



    /* Find a device number in the map to add or reuse the existing one */

    ndev = spapr_msicfg_find(phb, config_addr, true);

    if (ndev >= SPAPR_MSIX_MAX_DEVS || ndev < 0) {

        error_report("No free entry for a new MSI device");

        rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

        return;

    }

    trace_spapr_pci_msi("Configuring MSI", ndev, config_addr);



    /* Check if the device supports as many IRQs as requested */

    if (ret_intr_type == RTAS_TYPE_MSI) {

        max_irqs = msi_nr_vectors_allocated(pdev);

    } else if (ret_intr_type == RTAS_TYPE_MSIX) {

        max_irqs = pdev->msix_entries_nr;

    }

    if (!max_irqs) {

        error_report("Requested interrupt type %d is not enabled for device#%d",

                     ret_intr_type, ndev);

        rtas_st(rets, 0, -1); /* Hardware error */

        return;

    }

    /* Correct the number if the guest asked for too many */

    if (req_num > max_irqs) {

        req_num = max_irqs;

    }



    /* Check if there is an old config and MSI number has not changed */

    if (phb->msi_table[ndev].nvec && (req_num != phb->msi_table[ndev].nvec)) {

        /* Unexpected behaviour */

        error_report("Cannot reuse MSI config for device#%d", ndev);

        rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

        return;

    }



    /* There is no cached config, allocate MSIs */

    if (!phb->msi_table[ndev].nvec) {

        irq = xics_alloc_block(spapr->icp, 0, req_num, false,

                               ret_intr_type == RTAS_TYPE_MSI);

        if (irq < 0) {

            error_report("Cannot allocate MSIs for device#%d", ndev);

            rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

            return;

        }

        phb->msi_table[ndev].irq = irq;

        phb->msi_table[ndev].nvec = req_num;

        phb->msi_table[ndev].config_addr = config_addr;

    }



    /* Setup MSI/MSIX vectors in the device (via cfgspace or MSIX BAR) */

    spapr_msi_setmsg(pdev, spapr->msi_win_addr, ret_intr_type == RTAS_TYPE_MSIX,

                     phb->msi_table[ndev].irq, req_num);



    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

    rtas_st(rets, 1, req_num);

    rtas_st(rets, 2, ++seq_num);

    rtas_st(rets, 3, ret_intr_type);



    trace_spapr_pci_rtas_ibm_change_msi(func, req_num);

}
