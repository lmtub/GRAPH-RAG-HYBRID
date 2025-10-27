void object_property_set_link(Object *obj, Object *value,

                              const char *name, Error **errp)

{

    object_property_set_str(obj, object_get_canonical_path(value),

                            name, errp);

}
