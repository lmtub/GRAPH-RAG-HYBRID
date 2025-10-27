void qdev_set_parent_bus(DeviceState *dev, BusState *bus)

{

    Property *prop;



    if (qdev_hotplug) {

        assert(bus->allow_hotplug);

    }



    dev->parent_bus = bus;

    QTAILQ_INSERT_HEAD(&bus->children, dev, sibling);



    qdev_prop_set_defaults(dev, dev->parent_bus->info->props);

    for (prop = qdev_get_bus_info(dev)->props; prop && prop->name; prop++) {

        qdev_property_add_legacy(dev, prop, NULL);

        qdev_property_add_static(dev, prop, NULL);

    }

}
