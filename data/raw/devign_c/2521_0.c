static uint64_t bonito_pciconf_readl(void *opaque, target_phys_addr_t addr,

                                     unsigned size)

{



    PCIBonitoState *s = opaque;

    PCIDevice *d = PCI_DEVICE(s);



    DPRINTF("bonito_pciconf_readl "TARGET_FMT_plx"\n", addr);

    return d->config_read(d, addr, 4);

}
