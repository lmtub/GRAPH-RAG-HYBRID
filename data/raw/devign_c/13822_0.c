static void set_enum(Object *obj, Visitor *v, void *opaque,

                     const char *name, Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    Property *prop = opaque;

    int *ptr = qdev_get_prop_ptr(dev, prop);



    if (dev->state != DEV_STATE_CREATED) {

        error_set(errp, QERR_PERMISSION_DENIED);

        return;

    }



    visit_type_enum(v, ptr, prop->info->enum_table,

                    prop->info->name, prop->name, errp);

}
