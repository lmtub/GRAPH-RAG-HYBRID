static void set_chr(Object *obj, Visitor *v, const char *name, void *opaque,

                    Error **errp)

{

    DeviceState *dev = DEVICE(obj);

    Error *local_err = NULL;

    Property *prop = opaque;

    CharBackend *be = qdev_get_prop_ptr(dev, prop);

    CharDriverState *s;

    char *str;



    if (dev->realized) {

        qdev_prop_set_after_realize(dev, name, errp);

        return;

    }



    visit_type_str(v, name, &str, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    if (!*str) {

        g_free(str);

        be->chr = NULL;

        return;

    }



    s = qemu_chr_find(str);

    g_free(str);

    if (s == NULL) {

        error_setg(errp, "Property '%s.%s' can't find value '%s'",

                   object_get_typename(obj), prop->name, str);

        return;

    }



    if (!qemu_chr_fe_init(be, s, errp)) {

        error_prepend(errp, "Property '%s.%s' can't take value '%s': ",

                      object_get_typename(obj), prop->name, str);

        return;

    }

}
