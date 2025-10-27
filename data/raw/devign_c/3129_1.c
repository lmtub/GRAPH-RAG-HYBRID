static void pci_bridge_region_cleanup(PCIBridge *br)

{

    PCIBus *parent = br->dev.bus;

    pci_bridge_cleanup_alias(&br->alias_io,

                             parent->address_space_io);

    pci_bridge_cleanup_alias(&br->alias_mem,

                             parent->address_space_mem);

    pci_bridge_cleanup_alias(&br->alias_pref_mem,

                             parent->address_space_mem);

}
