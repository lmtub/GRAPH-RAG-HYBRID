static void ahci_irq_lower(AHCIState *s, AHCIDevice *dev)

{

    AHCIPCIState *d = container_of(s, AHCIPCIState, ahci);



    DPRINTF(0, "lower irq\n");



    if (!msi_enabled(PCI_DEVICE(d))) {

        qemu_irq_lower(s->irq);

    }

}
