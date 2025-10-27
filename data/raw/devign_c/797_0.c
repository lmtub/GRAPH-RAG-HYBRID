static void device_initfn(Object *obj)

{

    DeviceState *dev = DEVICE(obj);

    Property *prop;



    if (qdev_hotplug) {

        dev->hotplugged = 1;

        qdev_hot_added = true;

    }



    dev->instance_id_alias = -1;

    dev->state = DEV_STATE_CREATED;



    qdev_prop_set_defaults(dev, qdev_get_props(dev));

    for (prop = qdev_get_props(dev); prop && prop->name; prop++) {

        qdev_property_add_legacy(dev, prop, NULL);

        qdev_property_add_static(dev, prop, NULL);

    }



    object_property_add_str(OBJECT(dev), "type", qdev_get_type, NULL, NULL);

}
