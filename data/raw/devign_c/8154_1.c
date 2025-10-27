static void spapr_phb_placement(sPAPRMachineState *spapr, uint32_t index,

                                uint64_t *buid, hwaddr *pio,

                                hwaddr *mmio32, hwaddr *mmio64,

                                unsigned n_dma, uint32_t *liobns, Error **errp)

{

    /*

     * New-style PHB window placement.

     *

     * Goals: Gives large (1TiB), naturally aligned 64-bit MMIO window

     * for each PHB, in addition to 2GiB 32-bit MMIO and 64kiB PIO

     * windows.

     *

     * Some guest kernels can't work with MMIO windows above 1<<46

     * (64TiB), so we place up to 31 PHBs in the area 32TiB..64TiB

     *

     * 32TiB..(33TiB+1984kiB) contains the 64kiB PIO windows for each

     * PHB stacked together.  (32TiB+2GiB)..(32TiB+64GiB) contains the

     * 2GiB 32-bit MMIO windows for each PHB.  Then 33..64TiB has the

     * 1TiB 64-bit MMIO windows for each PHB.

     */

    const uint64_t base_buid = 0x800000020000000ULL;

    const int max_phbs =

        (SPAPR_PCI_LIMIT - SPAPR_PCI_BASE) / SPAPR_PCI_MEM64_WIN_SIZE - 1;

    int i;



    /* Sanity check natural alignments */

    QEMU_BUILD_BUG_ON((SPAPR_PCI_BASE % SPAPR_PCI_MEM64_WIN_SIZE) != 0);

    QEMU_BUILD_BUG_ON((SPAPR_PCI_LIMIT % SPAPR_PCI_MEM64_WIN_SIZE) != 0);

    QEMU_BUILD_BUG_ON((SPAPR_PCI_MEM64_WIN_SIZE % SPAPR_PCI_MEM32_WIN_SIZE) != 0);

    QEMU_BUILD_BUG_ON((SPAPR_PCI_MEM32_WIN_SIZE % SPAPR_PCI_IO_WIN_SIZE) != 0);

    /* Sanity check bounds */

    QEMU_BUILD_BUG_ON((max_phbs * SPAPR_PCI_IO_WIN_SIZE) > SPAPR_PCI_MEM32_WIN_SIZE);

    QEMU_BUILD_BUG_ON((max_phbs * SPAPR_PCI_MEM32_WIN_SIZE) > SPAPR_PCI_MEM64_WIN_SIZE);



    if (index >= max_phbs) {

        error_setg(errp, "\"index\" for PAPR PHB is too large (max %u)",

                   max_phbs - 1);

        return;

    }



    *buid = base_buid + index;

    for (i = 0; i < n_dma; ++i) {

        liobns[i] = SPAPR_PCI_LIOBN(index, i);

    }



    *pio = SPAPR_PCI_BASE + index * SPAPR_PCI_IO_WIN_SIZE;

    *mmio32 = SPAPR_PCI_BASE + (index + 1) * SPAPR_PCI_MEM32_WIN_SIZE;

    *mmio64 = SPAPR_PCI_BASE + (index + 1) * SPAPR_PCI_MEM64_WIN_SIZE;

}
