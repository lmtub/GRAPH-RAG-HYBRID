void memory_region_unregister_iommu_notifier(MemoryRegion *mr, Notifier *n)

{

    notifier_remove(n);

    if (mr->iommu_ops->notify_stopped &&

        QLIST_EMPTY(&mr->iommu_notify.notifiers)) {

        mr->iommu_ops->notify_stopped(mr);

    }

}
