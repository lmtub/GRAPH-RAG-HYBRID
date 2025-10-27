int qdev_unplug(DeviceState *dev)

{

    if (!dev->parent_bus->allow_hotplug) {

        qerror_report(QERR_BUS_NO_HOTPLUG, dev->parent_bus->name);

        return -1;

    }

    assert(dev->info->unplug != NULL);



    if (dev->ref != 0) {

        qerror_report(QERR_DEVICE_IN_USE, dev->id?:"");

        return -1;

    }



    qdev_hot_removed = true;



    return dev->info->unplug(dev);

}
