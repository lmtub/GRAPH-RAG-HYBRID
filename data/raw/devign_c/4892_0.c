void memory_region_iommu_replay_all(MemoryRegion *mr)

{

    IOMMUNotifier *notifier;



    IOMMU_NOTIFIER_FOREACH(notifier, mr) {

        memory_region_iommu_replay(mr, notifier, false);

    }

}
