static uint8_t qpci_pc_config_readb(QPCIBus *bus, int devfn, uint8_t offset)

{

    outl(0xcf8, (1 << 31) | (devfn << 8) | offset);

    return inb(0xcfc);

}
