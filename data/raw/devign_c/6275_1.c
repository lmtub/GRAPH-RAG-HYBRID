static void device_initfn(Object *obj)

{

    DeviceState *dev = DEVICE(obj);

    ObjectClass *class;

    Property *prop;



    if (qdev_hotplug) {

        dev->hotplugged = 1;

        qdev_hot_added = true;

    }



    dev->instance_id_alias = -1;

    dev->realized = false;



    object_property_add_bool(obj, "realized",

                             device_get_realized, device_set_realized, NULL);

    object_property_add_bool(obj, "hotpluggable",

                             device_get_hotpluggable, NULL, NULL);

    object_property_add_bool(obj, "hotplugged",

                             device_get_hotplugged, device_set_hotplugged,

                             &error_abort);



    class = object_get_class(OBJECT(dev));

    do {

        for (prop = DEVICE_CLASS(class)->props; prop && prop->name; prop++) {

            qdev_property_add_legacy(dev, prop, &error_abort);

            qdev_property_add_static(dev, prop, &error_abort);

        }

        class = object_class_get_parent(class);

    } while (class != object_class_by_name(TYPE_DEVICE));



    object_property_add_link(OBJECT(dev), "parent_bus", TYPE_BUS,

                             (Object **)&dev->parent_bus, NULL, 0,

                             &error_abort);

    QLIST_INIT(&dev->gpios);

}
