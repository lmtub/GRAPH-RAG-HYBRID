static uint16_t qvirtio_pci_config_readw(QVirtioDevice *d, uint64_t off)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    uint16_t value;



    value = qpci_io_readw(dev->pdev, CONFIG_BASE(dev) + off);

    if (qvirtio_is_big_endian(d)) {

        value = bswap16(value);

    }

    return value;

}
