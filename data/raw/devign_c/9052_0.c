static void pc_machine_device_post_plug_cb(HotplugHandler *hotplug_dev,

                                           DeviceState *dev, Error **errp)

{

    if (object_dynamic_cast(OBJECT(dev), TYPE_PC_DIMM)) {

        pc_dimm_post_plug(hotplug_dev, dev, errp);

    }

}
