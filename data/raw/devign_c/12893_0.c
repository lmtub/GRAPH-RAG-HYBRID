static void test_pci_enable(void)

{

    AHCIQState *ahci;



    ahci = ahci_boot();

    ahci_pci_enable(ahci);

    ahci_shutdown(ahci);

}
