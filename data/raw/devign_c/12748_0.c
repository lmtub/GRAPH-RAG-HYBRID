void memory_region_register_iommu_notifier(MemoryRegion *mr, Notifier *n)

{

    if (mr->iommu_ops->notify_started &&

        QLIST_EMPTY(&mr->iommu_notify.notifiers)) {

        mr->iommu_ops->notify_started(mr);

    }

    notifier_list_add(&mr->iommu_notify, n);

}
