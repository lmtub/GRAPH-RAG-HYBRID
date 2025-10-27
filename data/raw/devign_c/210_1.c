int msix_init_exclusive_bar(PCIDevice *dev, unsigned short nentries,

                            uint8_t bar_nr)

{

    int ret;

    char *name;

    uint32_t bar_size = 4096;

    uint32_t bar_pba_offset = bar_size / 2;

    uint32_t bar_pba_size = (nentries / 8 + 1) * 8;



    /*

     * Migration compatibility dictates that this remains a 4k

     * BAR with the vector table in the lower half and PBA in

     * the upper half for nentries which is lower or equal to 128.

     * No need to care about using more than 65 entries for legacy

     * machine types who has at most 64 queues.

     */

    if (nentries * PCI_MSIX_ENTRY_SIZE > bar_pba_offset) {

        bar_pba_offset = nentries * PCI_MSIX_ENTRY_SIZE;

    }



    if (bar_pba_offset + bar_pba_size > 4096) {

        bar_size = bar_pba_offset + bar_pba_size;

    }



    if (bar_size & (bar_size - 1)) {

        bar_size = 1 << qemu_fls(bar_size);

    }



    name = g_strdup_printf("%s-msix", dev->name);

    memory_region_init(&dev->msix_exclusive_bar, OBJECT(dev), name, bar_size);

    g_free(name);



    ret = msix_init(dev, nentries, &dev->msix_exclusive_bar, bar_nr,

                    0, &dev->msix_exclusive_bar,

                    bar_nr, bar_pba_offset,

                    0);

    if (ret) {

        return ret;

    }



    pci_register_bar(dev, bar_nr, PCI_BASE_ADDRESS_SPACE_MEMORY,

                     &dev->msix_exclusive_bar);



    return 0;

}
