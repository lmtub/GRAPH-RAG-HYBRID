void object_property_add_bool(Object *obj, const char *name,

                              bool (*get)(Object *, Error **),

                              void (*set)(Object *, bool, Error **),

                              Error **errp)

{

    BoolProperty *prop = g_malloc0(sizeof(*prop));



    prop->get = get;

    prop->set = set;



    object_property_add(obj, name, "bool",

                        get ? property_get_bool : NULL,

                        set ? property_set_bool : NULL,

                        property_release_bool,

                        prop, errp);

}
