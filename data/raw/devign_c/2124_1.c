DeviceState *qdev_device_add(QemuOpts *opts, Error **errp)

{

    DeviceClass *dc;

    const char *driver, *path;

    DeviceState *dev;

    BusState *bus = NULL;

    Error *err = NULL;



    driver = qemu_opt_get(opts, "driver");

    if (!driver) {

        error_setg(errp, QERR_MISSING_PARAMETER, "driver");

        return NULL;

    }



    /* find driver */

    dc = qdev_get_device_class(&driver, errp);

    if (!dc) {

        return NULL;

    }



    /* find bus */

    path = qemu_opt_get(opts, "bus");

    if (path != NULL) {

        bus = qbus_find(path, errp);

        if (!bus) {

            return NULL;

        }

        if (!object_dynamic_cast(OBJECT(bus), dc->bus_type)) {

            error_setg(errp, "Device '%s' can't go on %s bus",

                       driver, object_get_typename(OBJECT(bus)));

            return NULL;

        }

    } else if (dc->bus_type != NULL) {

        bus = qbus_find_recursive(sysbus_get_default(), NULL, dc->bus_type);

        if (!bus || qbus_is_full(bus)) {

            error_setg(errp, "No '%s' bus found for device '%s'",

                       dc->bus_type, driver);

            return NULL;

        }

    }

    if (qdev_hotplug && bus && !qbus_is_hotpluggable(bus)) {

        error_setg(errp, QERR_BUS_NO_HOTPLUG, bus->name);

        return NULL;

    }



    if (!migration_is_idle()) {

        error_setg(errp, "device_add not allowed while migrating");

        return NULL;

    }



    /* create device */

    dev = DEVICE(object_new(driver));



    if (bus) {

        qdev_set_parent_bus(dev, bus);






    }



    qdev_set_id(dev, qemu_opts_id(opts));



    /* set properties */

    if (qemu_opt_foreach(opts, set_property, dev, &err)) {


    }



    dev->opts = opts;

    object_property_set_bool(OBJECT(dev), true, "realized", &err);

    if (err != NULL) {

        dev->opts = NULL;


    }

    return dev;



err_del_dev:

    error_propagate(errp, err);

    object_unparent(OBJECT(dev));

    object_unref(OBJECT(dev));

    return NULL;

}