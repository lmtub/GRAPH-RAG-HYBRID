DeviceState *aux_create_slave(AUXBus *bus, const char *type, uint32_t addr)

{

    DeviceState *dev;



    dev = DEVICE(object_new(type));

    assert(dev);

    qdev_set_parent_bus(dev, &bus->qbus);

    qdev_init_nofail(dev);

    aux_bus_map_device(AUX_BUS(qdev_get_parent_bus(dev)), AUX_SLAVE(dev), addr);

    return dev;

}
