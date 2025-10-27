static void rtas_ibm_slot_error_detail(PowerPCCPU *cpu,

                                       sPAPREnvironment *spapr,

                                       uint32_t token, uint32_t nargs,

                                       target_ulong args, uint32_t nret,

                                       target_ulong rets)

{

    sPAPRPHBState *sphb;

    sPAPRPHBClass *spc;

    int option;

    uint64_t buid;



    if ((nargs != 8) || (nret != 1)) {

        goto param_error_exit;

    }



    buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    sphb = find_phb(spapr, buid);

    if (!sphb) {

        goto param_error_exit;

    }



    spc = SPAPR_PCI_HOST_BRIDGE_GET_CLASS(sphb);

    if (!spc->eeh_set_option) {

        goto param_error_exit;

    }



    option = rtas_ld(args, 7);

    switch (option) {

    case RTAS_SLOT_TEMP_ERR_LOG:

    case RTAS_SLOT_PERM_ERR_LOG:

        break;

    default:

        goto param_error_exit;

    }



    /* We don't have error log yet */

    rtas_st(rets, 0, RTAS_OUT_NO_ERRORS_FOUND);

    return;



param_error_exit:

    rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

}
