DeviceState *i2c_create_slave(i2c_bus *bus, const char *name, uint8_t addr)

{

    DeviceState *dev;



    dev = qdev_create(&bus->qbus, name);

    qdev_prop_set_uint8(dev, "address", addr);

    qdev_init(dev);

    return dev;

}
