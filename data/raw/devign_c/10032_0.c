static void vtd_address_space_unmap(VTDAddressSpace *as, IOMMUNotifier *n)

{

    IOMMUTLBEntry entry;

    hwaddr size;

    hwaddr start = n->start;

    hwaddr end = n->end;



    /*

     * Note: all the codes in this function has a assumption that IOVA

     * bits are no more than VTD_MGAW bits (which is restricted by

     * VT-d spec), otherwise we need to consider overflow of 64 bits.

     */



    if (end > VTD_ADDRESS_SIZE(VTD_HOST_ADDRESS_WIDTH)) {

        /*

         * Don't need to unmap regions that is bigger than the whole

         * VT-d supported address space size

         */

        end = VTD_ADDRESS_SIZE(VTD_HOST_ADDRESS_WIDTH);

    }



    assert(start <= end);

    size = end - start;



    if (ctpop64(size) != 1) {

        /*

         * This size cannot format a correct mask. Let's enlarge it to

         * suite the minimum available mask.

         */

        int n = 64 - clz64(size);

        if (n > VTD_MGAW) {

            /* should not happen, but in case it happens, limit it */

            n = VTD_MGAW;

        }

        size = 1ULL << n;

    }



    entry.target_as = &address_space_memory;

    /* Adjust iova for the size */

    entry.iova = n->start & ~(size - 1);

    /* This field is meaningless for unmap */

    entry.translated_addr = 0;

    entry.perm = IOMMU_NONE;

    entry.addr_mask = size - 1;



    trace_vtd_as_unmap_whole(pci_bus_num(as->bus),

                             VTD_PCI_SLOT(as->devfn),

                             VTD_PCI_FUNC(as->devfn),

                             entry.iova, size);



    memory_region_notify_one(n, &entry);

}
