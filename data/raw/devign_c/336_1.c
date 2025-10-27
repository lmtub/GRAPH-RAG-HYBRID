static void ahci_pci_enable(AHCIQState *ahci)

{

    uint8_t reg;



    start_ahci_device(ahci);



    switch (ahci->fingerprint) {

    case AHCI_INTEL_ICH9:

        /* ICH9 has a register at PCI 0x92 that

         * acts as a master port enabler mask. */

        reg = qpci_config_readb(ahci->dev, 0x92);

        reg |= 0x3F;

        qpci_config_writeb(ahci->dev, 0x92, reg);

        /* 0...0111111b -- bit significant, ports 0-5 enabled. */

        ASSERT_BIT_SET(qpci_config_readb(ahci->dev, 0x92), 0x3F);

        break;

    }



}
