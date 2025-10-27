static void stop_tco(const TestData *d)

{

    uint32_t val;



    val = qpci_io_readw(d->dev, d->tco_io_base + TCO1_CNT);

    val |= TCO_TMR_HLT;

    qpci_io_writew(d->dev, d->tco_io_base + TCO1_CNT, val);

}
