static void qdev_prop_set(DeviceState *dev, const char *name, void *src, enum PropertyType type)

{

    Property *prop;



    prop = qdev_prop_find(dev, name);

    if (!prop) {

        fprintf(stderr, "%s: property \"%s.%s\" not found\n",

                __FUNCTION__, object_get_typename(OBJECT(dev)), name);

        abort();

    }

    if (prop->info->type != type) {

        fprintf(stderr, "%s: property \"%s.%s\" type mismatch\n",

                __FUNCTION__, object_get_typename(OBJECT(dev)), name);

        abort();

    }

    qdev_prop_cpy(dev, prop, src);

}
