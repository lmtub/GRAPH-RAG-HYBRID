static void pci_ich9_ahci_realize(PCIDevice *dev, Error **errp)

{

    struct AHCIPCIState *d;

    int sata_cap_offset;

    uint8_t *sata_cap;

    d = ICH_AHCI(dev);



    ahci_realize(&d->ahci, DEVICE(dev), pci_get_address_space(dev), 6);



    pci_config_set_prog_interface(dev->config, AHCI_PROGMODE_MAJOR_REV_1);



    dev->config[PCI_CACHE_LINE_SIZE] = 0x08;  /* Cache line size */

    dev->config[PCI_LATENCY_TIMER]   = 0x00;  /* Latency timer */

    pci_config_set_interrupt_pin(dev->config, 1);



    /* XXX Software should program this register */

    dev->config[0x90]   = 1 << 6; /* Address Map Register - AHCI mode */



    d->ahci.irq = pci_allocate_irq(dev);



    pci_register_bar(dev, ICH9_IDP_BAR, PCI_BASE_ADDRESS_SPACE_IO,

                     &d->ahci.idp);

    pci_register_bar(dev, ICH9_MEM_BAR, PCI_BASE_ADDRESS_SPACE_MEMORY,

                     &d->ahci.mem);



    sata_cap_offset = pci_add_capability2(dev, PCI_CAP_ID_SATA,

                                          ICH9_SATA_CAP_OFFSET, SATA_CAP_SIZE,

                                          errp);

    if (sata_cap_offset < 0) {

        return;

    }



    sata_cap = dev->config + sata_cap_offset;

    pci_set_word(sata_cap + SATA_CAP_REV, 0x10);

    pci_set_long(sata_cap + SATA_CAP_BAR,

                 (ICH9_IDP_BAR + 0x4) | (ICH9_IDP_INDEX_LOG2 << 4));

    d->ahci.idp_offset = ICH9_IDP_INDEX;



    /* Although the AHCI 1.3 specification states that the first capability

     * should be PMCAP, the Intel ICH9 data sheet specifies that the ICH9

     * AHCI device puts the MSI capability first, pointing to 0x80. */

    msi_init(dev, ICH9_MSI_CAP_OFFSET, 1, true, false);

}
