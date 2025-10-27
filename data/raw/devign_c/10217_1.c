static void pci_ehci_config(void)

{

    /* hands over all ports from companion uhci to ehci */

    qpci_io_writew(ehci1.dev, ehci1.base + 0x60, 1);

}
