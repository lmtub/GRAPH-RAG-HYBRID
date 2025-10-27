static void test_flush(void)

{

    AHCIQState *ahci;



    ahci = ahci_boot_and_enable();

    ahci_test_flush(ahci);

    ahci_shutdown(ahci);

}
