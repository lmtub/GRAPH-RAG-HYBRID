static void rtas_ibm_read_pci_config(sPAPREnvironment *spapr,

                                     uint32_t token, uint32_t nargs,

                                     target_ulong args,

                                     uint32_t nret, target_ulong rets)

{

    uint64_t buid;

    uint32_t size, addr;



    if ((nargs != 4) || (nret != 2)) {

        rtas_st(rets, 0, -1);

        return;

    }



    buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    size = rtas_ld(args, 3);

    addr = rtas_ld(args, 0);



    finish_read_pci_config(spapr, buid, addr, size, rets);

}
