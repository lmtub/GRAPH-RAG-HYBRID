static void property_get_str(Object *obj, Visitor *v, void *opaque,

                             const char *name, Error **errp)

{

    StringProperty *prop = opaque;

    char *value;



    value = prop->get(obj, errp);

    if (value) {

        visit_type_str(v, &value, name, errp);

        g_free(value);

    }

}
