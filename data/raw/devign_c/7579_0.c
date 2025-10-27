DeviceState *qdev_try_create(BusState *bus, const char *name)

{

    DeviceState *dev;



    if (object_class_by_name(name) == NULL) {

        return NULL;

    }

    dev = DEVICE(object_new(name));

    if (!dev) {

        return NULL;

    }



    if (!bus) {

        bus = sysbus_get_default();

    }



    qdev_set_parent_bus(dev, bus);

    qdev_prop_set_globals(dev);



    return dev;

}
