static void check_exception(sPAPREnvironment *spapr,

                            uint32_t token, uint32_t nargs,

                            target_ulong args,

                            uint32_t nret, target_ulong rets)

{

    uint32_t mask, buf, len;

    uint64_t xinfo;



    if ((nargs < 6) || (nargs > 7) || nret != 1) {

        rtas_st(rets, 0, -3);

        return;

    }



    xinfo = rtas_ld(args, 1);

    mask = rtas_ld(args, 2);

    buf = rtas_ld(args, 4);

    len = rtas_ld(args, 5);

    if (nargs == 7) {

        xinfo |= (uint64_t)rtas_ld(args, 6) << 32;

    }



    if ((mask & EVENT_MASK_EPOW) && pending_epow) {

        if (sizeof(*pending_epow) < len) {

            len = sizeof(*pending_epow);

        }



        cpu_physical_memory_write(buf, pending_epow, len);

        g_free(pending_epow);

        pending_epow = NULL;

        rtas_st(rets, 0, 0);

    } else {

        rtas_st(rets, 0, 1);

    }

}
