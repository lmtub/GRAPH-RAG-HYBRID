static void unin_data_write(void *opaque, hwaddr addr,

                            uint64_t val, unsigned len)

{

    UNINState *s = opaque;

    PCIHostState *phb = PCI_HOST_BRIDGE(s);

    UNIN_DPRINTF("write addr %" TARGET_FMT_plx " len %d val %"PRIx64"\n",

                 addr, len, val);

    pci_data_write(phb->bus,

                   unin_get_config_reg(phb->config_reg, addr),

                   val, len);

}
