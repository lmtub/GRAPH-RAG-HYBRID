uint8_t qpci_io_readb(QPCIDevice *dev, void *data)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        return dev->bus->pio_readb(dev->bus, addr);

    } else {

        uint8_t val;

        dev->bus->memread(dev->bus, addr, &val, sizeof(val));

        return val;

    }

}
