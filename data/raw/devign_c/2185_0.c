void qdev_prop_set_globals_for_type(DeviceState *dev, const char *typename,

                                    Error **errp)

{

    GlobalProperty *prop;



    QTAILQ_FOREACH(prop, &global_props, next) {

        Error *err = NULL;



        if (strcmp(typename, prop->driver) != 0) {

            continue;

        }

        prop->not_used = false;

        object_property_parse(OBJECT(dev), prop->value, prop->property, &err);

        if (err != NULL) {

            error_propagate(errp, err);

            return;

        }

    }

}
