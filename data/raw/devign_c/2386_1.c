void qpci_io_writel(QPCIDevice *dev, void *data, uint32_t value)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        dev->bus->pio_writel(dev->bus, addr, value);

    } else {

        value = cpu_to_le32(value);

        dev->bus->memwrite(dev->bus, addr, &value, sizeof(value));

    }

}
