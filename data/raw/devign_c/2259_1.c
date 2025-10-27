static void rtas_set_tce_bypass(sPAPREnvironment *spapr, uint32_t token,

                                uint32_t nargs, target_ulong args,

                                uint32_t nret, target_ulong rets)

{

    VIOsPAPRBus *bus = spapr->vio_bus;

    VIOsPAPRDevice *dev;

    uint32_t unit, enable;



    if (nargs != 2) {

        rtas_st(rets, 0, -3);

        return;

    }

    unit = rtas_ld(args, 0);

    enable = rtas_ld(args, 1);

    dev = spapr_vio_find_by_reg(bus, unit);

    if (!dev) {

        rtas_st(rets, 0, -3);

        return;

    }

    if (enable) {

        dev->flags |= VIO_PAPR_FLAG_DMA_BYPASS;

    } else {

        dev->flags &= ~VIO_PAPR_FLAG_DMA_BYPASS;

    }



    rtas_st(rets, 0, 0);

}
