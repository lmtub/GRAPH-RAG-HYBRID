static void aux_bus_map_device(AUXBus *bus, AUXSlave *dev, hwaddr addr)

{

    memory_region_add_subregion(bus->aux_io, addr, dev->mmio);

}
