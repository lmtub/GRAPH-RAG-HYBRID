static void test_port(int port)

{

    struct qhc uhci;



    g_assert(port > 0);

    qusb_pci_init_one(qs->pcibus, &uhci, QPCI_DEVFN(0x1d, 0), 4);

    uhci_port_test(&uhci, port - 1, UHCI_PORT_CCS);


}