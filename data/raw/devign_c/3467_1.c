uint32_t qpci_io_readl(QPCIDevice *dev, void *data)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        return dev->bus->pio_readl(dev->bus, addr);

    } else {

        uint32_t val;

        dev->bus->memread(dev->bus, addr, &val, sizeof(val));

        return le32_to_cpu(val);

    }

}
