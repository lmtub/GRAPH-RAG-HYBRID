static void clear_tco_status(const TestData *d)

{

    qpci_io_writew(d->dev, d->tco_io_base + TCO1_STS, 0x0008);

    qpci_io_writew(d->dev, d->tco_io_base + TCO2_STS, 0x0002);

    qpci_io_writew(d->dev, d->tco_io_base + TCO2_STS, 0x0004);

}
