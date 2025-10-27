static void rtas_read_pci_config(sPAPREnvironment *spapr,

                                 uint32_t token, uint32_t nargs,

                                 target_ulong args,

                                 uint32_t nret, target_ulong rets)

{

    uint32_t val, size, addr;

    PCIDevice *dev = find_dev(spapr, 0, rtas_ld(args, 0));



    if (!dev) {

        rtas_st(rets, 0, -1);

        return;

    }

    size = rtas_ld(args, 1);

    addr = rtas_pci_cfgaddr(rtas_ld(args, 0));

    val = pci_default_read_config(dev, addr, size);

    rtas_st(rets, 0, 0);

    rtas_st(rets, 1, val);

}
