static MemoryRegionSection address_space_do_translate(AddressSpace *as,

                                                      hwaddr addr,

                                                      hwaddr *xlat,

                                                      hwaddr *plen,

                                                      bool is_write,

                                                      bool is_mmio)

{

    IOMMUTLBEntry iotlb;

    MemoryRegionSection *section;

    MemoryRegion *mr;



    for (;;) {

        AddressSpaceDispatch *d = atomic_rcu_read(&as->dispatch);

        section = address_space_translate_internal(d, addr, &addr, plen, is_mmio);

        mr = section->mr;



        if (!mr->iommu_ops) {

            break;

        }



        iotlb = mr->iommu_ops->translate(mr, addr, is_write);

        addr = ((iotlb.translated_addr & ~iotlb.addr_mask)

                | (addr & iotlb.addr_mask));

        *plen = MIN(*plen, (addr | iotlb.addr_mask) - addr + 1);

        if (!(iotlb.perm & (1 << is_write))) {

            goto translate_fail;

        }



        as = iotlb.target_as;

    }



    *xlat = addr;



    return *section;



translate_fail:

    return (MemoryRegionSection) { .mr = &io_mem_unassigned };

}
