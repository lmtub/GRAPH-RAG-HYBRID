static void rtas_system_reboot(sPAPREnvironment *spapr,

                               uint32_t token, uint32_t nargs,

                               target_ulong args,

                               uint32_t nret, target_ulong rets)

{

    if (nargs != 0 || nret != 1) {

        rtas_st(rets, 0, -3);

        return;

    }

    qemu_system_reset_request();

    rtas_st(rets, 0, 0);

}
