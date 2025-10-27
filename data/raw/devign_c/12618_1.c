static bool aux_bus_is_bridge(AUXBus *bus, DeviceState *dev)

{

    return (dev == DEVICE(bus->bridge));

}
