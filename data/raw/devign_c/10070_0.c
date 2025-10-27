static void vfio_iommu_map_notify(IOMMUNotifier *n, IOMMUTLBEntry *iotlb)

{

    VFIOGuestIOMMU *giommu = container_of(n, VFIOGuestIOMMU, n);

    VFIOContainer *container = giommu->container;

    hwaddr iova = iotlb->iova + giommu->iommu_offset;

    MemoryRegion *mr;

    hwaddr xlat;

    hwaddr len = iotlb->addr_mask + 1;

    void *vaddr;

    int ret;



    trace_vfio_iommu_map_notify(iotlb->perm == IOMMU_NONE ? "UNMAP" : "MAP",

                                iova, iova + iotlb->addr_mask);



    if (iotlb->target_as != &address_space_memory) {

        error_report("Wrong target AS \"%s\", only system memory is allowed",

                     iotlb->target_as->name ? iotlb->target_as->name : "none");

        return;

    }



    /*

     * The IOMMU TLB entry we have just covers translation through

     * this IOMMU to its immediate target.  We need to translate

     * it the rest of the way through to memory.

     */

    rcu_read_lock();

    mr = address_space_translate(&address_space_memory,

                                 iotlb->translated_addr,

                                 &xlat, &len, iotlb->perm & IOMMU_WO);

    if (!memory_region_is_ram(mr)) {

        error_report("iommu map to non memory area %"HWADDR_PRIx"",

                     xlat);

        goto out;

    }

    /*

     * Translation truncates length to the IOMMU page size,

     * check that it did not truncate too much.

     */

    if (len & iotlb->addr_mask) {

        error_report("iommu has granularity incompatible with target AS");

        goto out;

    }



    if ((iotlb->perm & IOMMU_RW) != IOMMU_NONE) {

        vaddr = memory_region_get_ram_ptr(mr) + xlat;

        ret = vfio_dma_map(container, iova,

                           iotlb->addr_mask + 1, vaddr,

                           !(iotlb->perm & IOMMU_WO) || mr->readonly);

        if (ret) {

            error_report("vfio_dma_map(%p, 0x%"HWADDR_PRIx", "

                         "0x%"HWADDR_PRIx", %p) = %d (%m)",

                         container, iova,

                         iotlb->addr_mask + 1, vaddr, ret);

        }

    } else {

        ret = vfio_dma_unmap(container, iova, iotlb->addr_mask + 1);

        if (ret) {

            error_report("vfio_dma_unmap(%p, 0x%"HWADDR_PRIx", "

                         "0x%"HWADDR_PRIx") = %d (%m)",

                         container, iova,

                         iotlb->addr_mask + 1, ret);

        }

    }

out:

    rcu_read_unlock();

}
