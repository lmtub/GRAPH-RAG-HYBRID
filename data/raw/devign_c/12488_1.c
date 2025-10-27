static void qobject_input_start_struct(Visitor *v, const char *name, void **obj,

                                       size_t size, Error **errp)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qobject_input_get_object(qiv, name, true, errp);



    if (obj) {

        *obj = NULL;

    }

    if (!qobj) {

        return;

    }

    if (qobject_type(qobj) != QTYPE_QDICT) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "QDict");

        return;

    }



    qobject_input_push(qiv, qobj, obj);



    if (obj) {

        *obj = g_malloc0(size);

    }

}
