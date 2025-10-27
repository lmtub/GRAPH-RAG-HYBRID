static void pci_init_bus_master(PCIDevice *pci_dev)

{

    AddressSpace *dma_as = pci_device_iommu_address_space(pci_dev);



    memory_region_init_alias(&pci_dev->bus_master_enable_region,

                             OBJECT(pci_dev), "bus master",

                             dma_as->root, 0, memory_region_size(dma_as->root));

    memory_region_set_enabled(&pci_dev->bus_master_enable_region, false);

    address_space_init(&pci_dev->bus_master_as,

                       &pci_dev->bus_master_enable_region, pci_dev->name);

}
