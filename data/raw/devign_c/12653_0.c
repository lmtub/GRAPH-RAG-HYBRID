static void qmp_deserialize(void **native_out, void *datap,

                            VisitorFunc visit, Error **errp)

{

    QmpSerializeData *d = datap;

    QString *output_json;

    QObject *obj_orig, *obj;



    obj_orig = qmp_output_get_qobject(d->qov);

    output_json = qobject_to_json(obj_orig);

    obj = qobject_from_json(qstring_get_str(output_json));



    QDECREF(output_json);

    d->qiv = qmp_input_visitor_new(obj, true);

    qobject_decref(obj_orig);

    qobject_decref(obj);

    visit(d->qiv, native_out, errp);

}
