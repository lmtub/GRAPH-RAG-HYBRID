static void ehci_port_test(struct qhc *hc, int port, uint32_t expect)

{

    void *addr = hc->base + 0x64 + 4 * port;

    uint32_t value = qpci_io_readl(hc->dev, addr);

    uint16_t mask = ~(PORTSC_CSC | PORTSC_PEDC | PORTSC_OCC);



#if 0

    fprintf(stderr, "%s: %d, have 0x%08x, want 0x%08x\n",

            __func__, port, value & mask, expect & mask);

#endif

    g_assert((value & mask) == (expect & mask));

}
