static void vhost_iommu_region_add(MemoryListener *listener,

                                   MemoryRegionSection *section)

{

    struct vhost_dev *dev = container_of(listener, struct vhost_dev,

                                         iommu_listener);

    struct vhost_iommu *iommu;



    if (!memory_region_is_iommu(section->mr)) {

        return;

    }



    iommu = g_malloc0(sizeof(*iommu));

    iommu->n.notify = vhost_iommu_unmap_notify;

    iommu->n.notifier_flags = IOMMU_NOTIFIER_UNMAP;

    iommu->mr = section->mr;

    iommu->iommu_offset = section->offset_within_address_space -

                          section->offset_within_region;

    iommu->hdev = dev;

    memory_region_register_iommu_notifier(section->mr, &iommu->n);

    QLIST_INSERT_HEAD(&dev->iommu_list, iommu, iommu_next);

    /* TODO: can replay help performance here? */

}
