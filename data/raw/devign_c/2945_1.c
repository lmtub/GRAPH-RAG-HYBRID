int qdev_prop_check_globals(void)

{

    GlobalProperty *prop;

    int ret = 0;



    QTAILQ_FOREACH(prop, &global_props, next) {

        ObjectClass *oc;

        DeviceClass *dc;

        if (prop->used) {

            continue;

        }

        if (!prop->user_provided) {

            continue;

        }

        oc = object_class_by_name(prop->driver);

        oc = object_class_dynamic_cast(oc, TYPE_DEVICE);

        if (!oc) {

            error_report("Warning: global %s.%s has invalid class name",

                       prop->driver, prop->property);

            ret = 1;

            continue;

        }

        dc = DEVICE_CLASS(oc);

        if (!dc->hotpluggable && !prop->used) {

            error_report("Warning: global %s.%s=%s not used",

                       prop->driver, prop->property, prop->value);

            ret = 1;

            continue;

        }

    }

    return ret;

}
