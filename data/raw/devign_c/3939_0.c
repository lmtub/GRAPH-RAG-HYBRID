QObject *object_property_get_qobject(Object *obj, const char *name,

                                     Error **errp)

{

    QObject *ret = NULL;

    Error *local_err = NULL;

    Visitor *v;



    v = qmp_output_visitor_new(&ret);

    object_property_get(obj, v, name, &local_err);

    if (!local_err) {

        visit_complete(v, &ret);

    }

    error_propagate(errp, local_err);

    visit_free(v);

    return ret;

}
