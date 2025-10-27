int msix_init_exclusive_bar(PCIDevice *dev, unsigned short nentries,

                            uint8_t bar_nr)

{

    int ret;

    char *name;



    /*

     * Migration compatibility dictates that this remains a 4k

     * BAR with the vector table in the lower half and PBA in

     * the upper half.  Do not use these elsewhere!

     */

#define MSIX_EXCLUSIVE_BAR_SIZE 4096

#define MSIX_EXCLUSIVE_BAR_TABLE_OFFSET 0

#define MSIX_EXCLUSIVE_BAR_PBA_OFFSET (MSIX_EXCLUSIVE_BAR_SIZE / 2)

#define MSIX_EXCLUSIVE_CAP_OFFSET 0



    if (nentries * PCI_MSIX_ENTRY_SIZE > MSIX_EXCLUSIVE_BAR_PBA_OFFSET) {

        return -EINVAL;

    }



    name = g_strdup_printf("%s-msix", dev->name);

    memory_region_init(&dev->msix_exclusive_bar, OBJECT(dev), name, MSIX_EXCLUSIVE_BAR_SIZE);

    g_free(name);



    ret = msix_init(dev, nentries, &dev->msix_exclusive_bar, bar_nr,

                    MSIX_EXCLUSIVE_BAR_TABLE_OFFSET, &dev->msix_exclusive_bar,

                    bar_nr, MSIX_EXCLUSIVE_BAR_PBA_OFFSET,

                    MSIX_EXCLUSIVE_CAP_OFFSET);

    if (ret) {

        return ret;

    }



    pci_register_bar(dev, bar_nr, PCI_BASE_ADDRESS_SPACE_MEMORY,

                     &dev->msix_exclusive_bar);



    return 0;

}
