void s390_pci_iommu_enable(S390PCIBusDevice *pbdev)

{

    memory_region_init_iommu(&pbdev->iommu_mr, OBJECT(&pbdev->mr),

                             &s390_iommu_ops, "iommu-s390", pbdev->pal + 1);

    memory_region_add_subregion(&pbdev->mr, 0, &pbdev->iommu_mr);

    pbdev->iommu_enabled = true;

}
