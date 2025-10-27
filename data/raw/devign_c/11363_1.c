void object_property_add_link(Object *obj, const char *name,

                              const char *type, Object **child,

                              Error **errp)

{

    gchar *full_type;



    full_type = g_strdup_printf("link<%s>", type);



    object_property_add(obj, name, full_type,

                        object_get_link_property,

                        object_set_link_property,

                        NULL, child, errp);



    g_free(full_type);

}
