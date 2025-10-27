static uint64_t pci_host_data_read(void *opaque,

                                   hwaddr addr, unsigned len)

{

    PCIHostState *s = opaque;

    uint32_t val;

    if (!(s->config_reg & (1 << 31)))

        return 0xffffffff;

    val = pci_data_read(s->bus, s->config_reg | (addr & 3), len);

    PCI_DPRINTF("read addr " TARGET_FMT_plx " len %d val %x\n",

                addr, len, val);

    return val;

}
