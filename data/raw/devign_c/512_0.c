static void rtas_display_character(sPAPREnvironment *spapr,

                                   uint32_t token, uint32_t nargs,

                                   target_ulong args,

                                   uint32_t nret, target_ulong rets)

{

    uint8_t c = rtas_ld(args, 0);

    VIOsPAPRDevice *sdev = vty_lookup(spapr, 0);



    if (!sdev) {

        rtas_st(rets, 0, -1);

    } else {

        vty_putchars(sdev, &c, sizeof(c));

        rtas_st(rets, 0, 0);

    }

}
