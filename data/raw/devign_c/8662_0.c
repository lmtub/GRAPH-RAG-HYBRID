static IOMMUTLBEntry s390_translate_iommu(MemoryRegion *mr, hwaddr addr,

                                          bool is_write)

{

    uint64_t pte;

    uint32_t flags;

    S390PCIIOMMU *iommu = container_of(mr, S390PCIIOMMU, iommu_mr);

    IOMMUTLBEntry ret = {

        .target_as = &address_space_memory,

        .iova = 0,

        .translated_addr = 0,

        .addr_mask = ~(hwaddr)0,

        .perm = IOMMU_NONE,

    };



    switch (iommu->pbdev->state) {

    case ZPCI_FS_ENABLED:

    case ZPCI_FS_BLOCKED:

        if (!iommu->enabled) {

            return ret;

        }

        break;

    default:

        return ret;

    }



    DPRINTF("iommu trans addr 0x%" PRIx64 "\n", addr);



    if (addr < iommu->pba || addr > iommu->pal) {

        return ret;

    }



    pte = s390_guest_io_table_walk(s390_pci_get_table_origin(iommu->g_iota),

                                   addr);

    if (!pte) {

        return ret;

    }



    flags = pte & ZPCI_PTE_FLAG_MASK;

    ret.iova = addr;

    ret.translated_addr = pte & ZPCI_PTE_ADDR_MASK;

    ret.addr_mask = 0xfff;



    if (flags & ZPCI_PTE_INVALID) {

        ret.perm = IOMMU_NONE;

    } else {

        ret.perm = IOMMU_RW;

    }



    return ret;

}
