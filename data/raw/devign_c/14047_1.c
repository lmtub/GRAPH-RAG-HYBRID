static void rtas_ibm_query_interrupt_source_number(PowerPCCPU *cpu,

                                                   sPAPREnvironment *spapr,

                                                   uint32_t token,

                                                   uint32_t nargs,

                                                   target_ulong args,

                                                   uint32_t nret,

                                                   target_ulong rets)

{

    uint32_t config_addr = rtas_ld(args, 0);

    uint64_t buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    unsigned int intr_src_num = -1, ioa_intr_num = rtas_ld(args, 3);

    int ndev;

    sPAPRPHBState *phb = NULL;



    /* Fins sPAPRPHBState */

    phb = find_phb(spapr, buid);

    if (!phb) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    /* Find device descriptor and start IRQ */

    ndev = spapr_msicfg_find(phb, config_addr, false);

    if (ndev < 0) {

        trace_spapr_pci_msi("MSI has not been enabled", -1, config_addr);

        rtas_st(rets, 0, RTAS_OUT_HW_ERROR);

        return;

    }



    intr_src_num = phb->msi_table[ndev].irq + ioa_intr_num;

    trace_spapr_pci_rtas_ibm_query_interrupt_source_number(ioa_intr_num,

                                                           intr_src_num);



    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

    rtas_st(rets, 1, intr_src_num);

    rtas_st(rets, 2, 1);/* 0 == level; 1 == edge */

}
