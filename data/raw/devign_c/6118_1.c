static void qobject_input_type_null(Visitor *v, const char *name, Error **errp)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qobject_input_get_object(qiv, name, true, errp);



    if (!qobj) {

        return;

    }



    if (qobject_type(qobj) != QTYPE_QNULL) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "null");

    }

}
