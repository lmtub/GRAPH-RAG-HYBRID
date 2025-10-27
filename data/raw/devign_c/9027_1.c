void uhci_port_test(struct qhc *hc, int port, uint16_t expect)

{

    void *addr = hc->base + 0x10 + 2 * port;

    uint16_t value = qpci_io_readw(hc->dev, addr);

    uint16_t mask = ~(UHCI_PORT_WRITE_CLEAR | UHCI_PORT_RSVD1);



    g_assert((value & mask) == (expect & mask));

}
