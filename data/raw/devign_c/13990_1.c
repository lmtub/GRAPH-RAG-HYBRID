void qpci_io_writeq(QPCIDevice *dev, void *data, uint64_t value)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        dev->bus->pio_writeq(dev->bus, addr, value);

    } else {

        value = cpu_to_le64(value);

        dev->bus->memwrite(dev->bus, addr, &value, sizeof(value));

    }

}
