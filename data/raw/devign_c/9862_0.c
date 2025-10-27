static void *build_pci_bus_begin(PCIBus *bus, void *parent_state)

{

    AcpiBuildPciBusHotplugState *parent = parent_state;

    AcpiBuildPciBusHotplugState *child = g_malloc(sizeof *child);



    build_pci_bus_state_init(child, parent, parent->pcihp_bridge_en);



    return child;

}
