static QPCIDevice *get_ahci_device(uint32_t *fingerprint)

{

    QPCIDevice *ahci;

    uint32_t ahci_fingerprint;

    QPCIBus *pcibus;



    pcibus = qpci_init_pc();



    /* Find the AHCI PCI device and verify it's the right one. */

    ahci = qpci_device_find(pcibus, QPCI_DEVFN(0x1F, 0x02));

    g_assert(ahci != NULL);



    ahci_fingerprint = qpci_config_readl(ahci, PCI_VENDOR_ID);



    switch (ahci_fingerprint) {

    case AHCI_INTEL_ICH9:

        break;

    default:

        /* Unknown device. */

        g_assert_not_reached();

    }



    if (fingerprint) {

        *fingerprint = ahci_fingerprint;

    }

    return ahci;

}
