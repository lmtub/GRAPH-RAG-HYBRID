static void qdev_set_legacy_property(DeviceState *dev, Visitor *v, void *opaque,

                                     const char *name, Error **errp)

{

    Property *prop = opaque;



    if (dev->state != DEV_STATE_CREATED) {

        error_set(errp, QERR_PERMISSION_DENIED);

        return;

    }



    if (prop->info->parse) {

        Error *local_err = NULL;

        char *ptr = NULL;



        visit_type_str(v, &ptr, name, &local_err);

        if (!local_err) {

            int ret;

            ret = prop->info->parse(dev, prop, ptr);

            if (ret != 0) {

                error_set(errp, QERR_INVALID_PARAMETER_VALUE,

                          name, prop->info->name);

            }

            g_free(ptr);

        } else {

            error_propagate(errp, local_err);

        }

    } else {

        error_set(errp, QERR_PERMISSION_DENIED);

    }

}
