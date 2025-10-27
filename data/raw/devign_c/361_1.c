DeviceState *ssi_create_slave(SSIBus *bus, const char *name)

{

    DeviceState *dev;

    dev = qdev_create(&bus->qbus, name);

    qdev_init(dev);

    return dev;

}
