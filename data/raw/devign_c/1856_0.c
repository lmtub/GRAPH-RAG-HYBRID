static void qdev_get_legacy_property(DeviceState *dev, Visitor *v, void *opaque,

                                     const char *name, Error **errp)

{

    Property *prop = opaque;



    if (prop->info->print) {

        char buffer[1024];

        char *ptr = buffer;



        prop->info->print(dev, prop, buffer, sizeof(buffer));

        visit_type_str(v, &ptr, name, errp);

    } else {

        error_set(errp, QERR_PERMISSION_DENIED);

    }

}
