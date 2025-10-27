static void property_get_bool(Object *obj, Visitor *v, void *opaque,

                              const char *name, Error **errp)

{

    BoolProperty *prop = opaque;

    bool value;



    value = prop->get(obj, errp);

    visit_type_bool(v, &value, name, errp);

}
