static IOMMUTLBEntry pbm_translate_iommu(MemoryRegion *iommu, hwaddr addr,

                                         bool is_write)

{

    IOMMUState *is = container_of(iommu, IOMMUState, iommu);

    hwaddr baseaddr, offset;

    uint64_t tte;

    uint32_t tsbsize;

    IOMMUTLBEntry ret = {

        .target_as = &address_space_memory,

        .iova = 0,

        .translated_addr = 0,

        .addr_mask = ~(hwaddr)0,

        .perm = IOMMU_NONE,

    };



    if (!(is->regs[IOMMU_CTRL >> 3] & IOMMU_CTRL_MMU_EN)) {

        /* IOMMU disabled, passthrough using standard 8K page */

        ret.iova = addr & IOMMU_PAGE_MASK_8K;

        ret.translated_addr = addr;

        ret.addr_mask = IOMMU_PAGE_MASK_8K;

        ret.perm = IOMMU_RW;



        return ret;

    }



    baseaddr = is->regs[IOMMU_BASE >> 3];

    tsbsize = (is->regs[IOMMU_CTRL >> 3] >> IOMMU_CTRL_TSB_SHIFT) & 0x7;



    if (is->regs[IOMMU_CTRL >> 3] & IOMMU_CTRL_TBW_SIZE) {

        /* 64K */

        switch (tsbsize) {

        case 0:

            offset = (addr & IOMMU_TSB_64K_OFFSET_MASK_64M) >> 13;

            break;

        case 1:

            offset = (addr & IOMMU_TSB_64K_OFFSET_MASK_128M) >> 13;

            break;

        case 2:

            offset = (addr & IOMMU_TSB_64K_OFFSET_MASK_256M) >> 13;

            break;

        case 3:

            offset = (addr & IOMMU_TSB_64K_OFFSET_MASK_512M) >> 13;

            break;

        case 4:

            offset = (addr & IOMMU_TSB_64K_OFFSET_MASK_1G) >> 13;

            break;

        case 5:

            offset = (addr & IOMMU_TSB_64K_OFFSET_MASK_2G) >> 13;

            break;

        default:

            /* Not implemented, error */

            return ret;

        }

    } else {

        /* 8K */

        switch (tsbsize) {

        case 0:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_8M) >> 10;

            break;

        case 1:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_16M) >> 10;

            break;

        case 2:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_32M) >> 10;

            break;

        case 3:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_64M) >> 10;

            break;

        case 4:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_128M) >> 10;

            break;

        case 5:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_256M) >> 10;

            break;

        case 6:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_512M) >> 10;

            break;

        case 7:

            offset = (addr & IOMMU_TSB_8K_OFFSET_MASK_1G) >> 10;

            break;

        }

    }



    tte = address_space_ldq_be(&address_space_memory, baseaddr + offset,

                               MEMTXATTRS_UNSPECIFIED, NULL);



    if (!(tte & IOMMU_TTE_DATA_V)) {

        /* Invalid mapping */

        return ret;

    }



    if (tte & IOMMU_TTE_DATA_W) {

        /* Writeable */

        ret.perm = IOMMU_RW;

    } else {

        ret.perm = IOMMU_RO;

    }



    /* Extract phys */

    if (tte & IOMMU_TTE_DATA_SIZE) {

        /* 64K */

        ret.iova = addr & IOMMU_PAGE_MASK_64K;

        ret.translated_addr = tte & IOMMU_TTE_PHYS_MASK_64K;

        ret.addr_mask = (IOMMU_PAGE_SIZE_64K - 1);

    } else {

        /* 8K */

        ret.iova = addr & IOMMU_PAGE_MASK_8K;

        ret.translated_addr = tte & IOMMU_TTE_PHYS_MASK_8K;

        ret.addr_mask = (IOMMU_PAGE_SIZE_8K - 1);

    }



    return ret;

}
