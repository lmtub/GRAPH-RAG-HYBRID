static void pci_bridge_region_del(PCIBridge *br, PCIBridgeWindows *w)

{

    PCIDevice *pd = PCI_DEVICE(br);

    PCIBus *parent = pd->bus;



    memory_region_del_subregion(parent->address_space_io, &w->alias_io);

    memory_region_del_subregion(parent->address_space_mem, &w->alias_mem);

    memory_region_del_subregion(parent->address_space_mem, &w->alias_pref_mem);

    pci_unregister_vga(pd);

}
