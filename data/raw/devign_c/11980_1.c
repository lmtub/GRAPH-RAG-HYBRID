static void qpci_pc_config_writew(QPCIBus *bus, int devfn, uint8_t offset, uint16_t value)

{

    outl(0xcf8, (1 << 31) | (devfn << 8) | offset);

    outw(0xcfc, value);

}
