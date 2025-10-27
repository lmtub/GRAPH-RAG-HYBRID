static void vtd_iommu_notify_flag_changed(MemoryRegion *iommu,

                                          IOMMUNotifierFlag old,

                                          IOMMUNotifierFlag new)

{

    VTDAddressSpace *vtd_as = container_of(iommu, VTDAddressSpace, iommu);



    if (new & IOMMU_NOTIFIER_MAP) {

        error_report("Device at bus %s addr %02x.%d requires iommu "

                     "notifier which is currently not supported by "

                     "intel-iommu emulation",

                     vtd_as->bus->qbus.name, PCI_SLOT(vtd_as->devfn),

                     PCI_FUNC(vtd_as->devfn));

        exit(1);

    }

}
