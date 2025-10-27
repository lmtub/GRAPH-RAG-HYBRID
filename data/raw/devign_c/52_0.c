void acpi_memory_plug_cb(HotplugHandler *hotplug_dev, MemHotplugState *mem_st,

                         DeviceState *dev, Error **errp)

{

    MemStatus *mdev;

    DeviceClass *dc = DEVICE_GET_CLASS(dev);



    if (!dc->hotpluggable) {

        return;

    }



    mdev = acpi_memory_slot_status(mem_st, dev, errp);

    if (!mdev) {

        return;

    }



    mdev->dimm = dev;

    mdev->is_enabled = true;

    if (dev->hotplugged) {

        mdev->is_inserting = true;

        acpi_send_event(DEVICE(hotplug_dev), ACPI_MEMORY_HOTPLUG_STATUS);

    }

}
