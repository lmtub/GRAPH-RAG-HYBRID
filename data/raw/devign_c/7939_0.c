void object_property_add_link(Object *obj, const char *name,

                              const char *type, Object **child,

                              void (*check)(Object *, const char *,

                                            Object *, Error **),

                              ObjectPropertyLinkFlags flags,

                              Error **errp)

{

    Error *local_err = NULL;

    LinkProperty *prop = g_malloc(sizeof(*prop));

    gchar *full_type;



    prop->child = child;

    prop->check = check;

    prop->flags = flags;



    full_type = g_strdup_printf("link<%s>", type);



    object_property_add(obj, name, full_type,

                        object_get_link_property,

                        check ? object_set_link_property : NULL,

                        object_release_link_property,

                        prop,

                        &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        g_free(prop);

    }



    g_free(full_type);

}
