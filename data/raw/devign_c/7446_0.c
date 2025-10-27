static const char *pxb_host_root_bus_path(PCIHostState *host_bridge,

                                          PCIBus *rootbus)

{

    PXBBus *bus = PXB_BUS(rootbus);



    snprintf(bus->bus_path, 8, "0000:%02x", pxb_bus_num(rootbus));

    return bus->bus_path;

}
