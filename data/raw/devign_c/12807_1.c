void qpci_io_writeb(QPCIDevice *dev, void *data, uint8_t value)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        dev->bus->pio_writeb(dev->bus, addr, value);

    } else {

        dev->bus->memwrite(dev->bus, addr, &value, sizeof(value));

    }

}
