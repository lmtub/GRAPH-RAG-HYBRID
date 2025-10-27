void start_ahci_device(AHCIQState *ahci)

{

    /* Map AHCI's ABAR (BAR5) */

    ahci->hba_base = qpci_iomap(ahci->dev, 5, &ahci->barsize);

    g_assert(ahci->hba_base);



    /* turns on pci.cmd.iose, pci.cmd.mse and pci.cmd.bme */

    qpci_device_enable(ahci->dev);

}
