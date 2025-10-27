static target_ulong h_put_tce(CPUPPCState *env, sPAPREnvironment *spapr,

                              target_ulong opcode, target_ulong *args)

{

    target_ulong liobn = args[0];

    target_ulong ioba = args[1];

    target_ulong tce = args[2];

    VIOsPAPRDevice *dev = spapr_vio_find_by_reg(spapr->vio_bus, liobn);

    VIOsPAPR_RTCE *rtce;



    if (!dev) {

        hcall_dprintf("LIOBN 0x" TARGET_FMT_lx " does not exist\n", liobn);

        return H_PARAMETER;

    }



    ioba &= ~(SPAPR_VIO_TCE_PAGE_SIZE - 1);



#ifdef DEBUG_TCE

    fprintf(stderr, "spapr_vio_put_tce on %s  ioba 0x" TARGET_FMT_lx

            "  TCE 0x" TARGET_FMT_lx "\n", dev->qdev.id, ioba, tce);

#endif



    if (ioba >= dev->rtce_window_size) {

        hcall_dprintf("Out-of-bounds IOBA 0x" TARGET_FMT_lx "\n", ioba);

        return H_PARAMETER;

    }



    rtce = dev->rtce_table + (ioba >> SPAPR_VIO_TCE_PAGE_SHIFT);

    rtce->tce = tce;



    return H_SUCCESS;

}
