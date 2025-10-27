uint16_t qpci_io_readw(QPCIDevice *dev, void *data)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        return dev->bus->pio_readw(dev->bus, addr);

    } else {

        uint16_t val;

        dev->bus->memread(dev->bus, addr, &val, sizeof(val));

        return le16_to_cpu(val);

    }

}
