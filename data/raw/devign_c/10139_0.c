static void build_pci_bus_state_init(AcpiBuildPciBusHotplugState *state,

                                     AcpiBuildPciBusHotplugState *parent,

                                     bool pcihp_bridge_en)

{

    state->parent = parent;

    state->device_table = build_alloc_array();

    state->notify_table = build_alloc_array();

    state->pcihp_bridge_en = pcihp_bridge_en;

}
