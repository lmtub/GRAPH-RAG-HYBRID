static void pc_xen_hvm_init(MachineState *machine)
{
    PCIBus *bus;
    pc_xen_hvm_init_pci(machine);
    bus = pci_find_primary_bus();
    if (bus != NULL) {
        pci_create_simple(bus, -1, "xen-platform");