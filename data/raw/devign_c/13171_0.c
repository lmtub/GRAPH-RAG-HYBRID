static void rtas_get_xive(sPAPREnvironment *spapr, uint32_t token,

                          uint32_t nargs, target_ulong args,

                          uint32_t nret, target_ulong rets)

{

    struct ics_state *ics = spapr->icp->ics;

    uint32_t nr;



    if ((nargs != 1) || (nret != 3)) {

        rtas_st(rets, 0, -3);

        return;

    }



    nr = rtas_ld(args, 0);



    if (!ics_valid_irq(ics, nr)) {

        rtas_st(rets, 0, -3);

        return;

    }



    rtas_st(rets, 0, 0); /* Success */

    rtas_st(rets, 1, ics->irqs[nr - ics->offset].server);

    rtas_st(rets, 2, ics->irqs[nr - ics->offset].priority);

}
