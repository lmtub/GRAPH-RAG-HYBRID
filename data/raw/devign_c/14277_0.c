static void device_unparent(Object *obj)

{

    DeviceState *dev = DEVICE(obj);



    if (dev->parent_bus != NULL) {

        bus_remove_child(dev->parent_bus, dev);

    }

}
