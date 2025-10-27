static void rtas_ibm_read_pci_config(sPAPREnvironment *spapr,

                                     uint32_t token, uint32_t nargs,

                                     target_ulong args,

                                     uint32_t nret, target_ulong rets)

{

    uint32_t val, size, addr;

    uint64_t buid = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 2);

    PCIDevice *dev = find_dev(spapr, buid, rtas_ld(args, 0));



    if (!dev) {

        rtas_st(rets, 0, -1);

        return;

    }

    size = rtas_ld(args, 3);

    addr = rtas_pci_cfgaddr(rtas_ld(args, 0));

    val = pci_default_read_config(dev, addr, size);

    rtas_st(rets, 0, 0);

    rtas_st(rets, 1, val);

}
