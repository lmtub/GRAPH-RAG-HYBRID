void object_property_set_qobject(Object *obj, QObject *value,

                                 const char *name, Error **errp)

{

    Visitor *v;

    /* TODO: Should we reject, rather than ignore, excess input? */

    v = qobject_input_visitor_new(value, false);

    object_property_set(obj, v, name, errp);

    visit_free(v);

}
