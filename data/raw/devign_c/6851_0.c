static void test_io_rw_interface(enum AddrMode lba48, enum IOMode dma,

                                 unsigned bufsize, uint64_t sector)

{

    AHCIQState *ahci;



    ahci = ahci_boot_and_enable();

    ahci_test_io_rw_simple(ahci, bufsize, sector,

                           io_cmds[dma][lba48][IO_READ],

                           io_cmds[dma][lba48][IO_WRITE]);

    ahci_shutdown(ahci);

}
