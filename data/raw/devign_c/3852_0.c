static QPCIDevice *start_ahci_device(QPCIDevice *ahci, void **hba_base)

{

    /* Map AHCI's ABAR (BAR5) */

    *hba_base = qpci_iomap(ahci, 5, NULL);



    /* turns on pci.cmd.iose, pci.cmd.mse and pci.cmd.bme */

    qpci_device_enable(ahci);



    return ahci;

}
