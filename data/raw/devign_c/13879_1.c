static void qpci_pc_config_writeb(QPCIBus *bus, int devfn, uint8_t offset, uint8_t value)

{

    outl(0xcf8, (1 << 31) | (devfn << 8) | offset);

    outb(0xcfc, value);

}
