void qdev_property_add_child(DeviceState *dev, const char *name,

                             DeviceState *child, Error **errp)

{

    gchar *type;



    type = g_strdup_printf("child<%s>", child->info->name);



    qdev_property_add(dev, name, type, qdev_get_child_property,

                      NULL, NULL, child, errp);



    qdev_ref(child);

    g_assert(child->parent == NULL);

    child->parent = dev;



    g_free(type);

}
