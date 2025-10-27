static void spapr_core_release(DeviceState *dev, void *opaque)

{

    HotplugHandler *hotplug_ctrl;



    hotplug_ctrl = qdev_get_hotplug_handler(dev);

    hotplug_handler_unplug(hotplug_ctrl, dev, &error_abort);

}
