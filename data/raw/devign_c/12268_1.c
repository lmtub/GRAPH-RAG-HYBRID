static CCIDBus *ccid_bus_new(DeviceState *dev)

{

    CCIDBus *bus;



    bus = FROM_QBUS(CCIDBus, qbus_create(&ccid_bus_info, dev, NULL));

    bus->qbus.allow_hotplug = 1;



    return bus;

}
