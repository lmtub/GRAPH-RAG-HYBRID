static void bonito_pciconf_writel(void *opaque, target_phys_addr_t addr,

                                  uint64_t val, unsigned size)

{

    PCIBonitoState *s = opaque;

    PCIDevice *d = PCI_DEVICE(s);



    DPRINTF("bonito_pciconf_writel "TARGET_FMT_plx" val %x\n", addr, val);

    d->config_write(d, addr, val, 4);

}
