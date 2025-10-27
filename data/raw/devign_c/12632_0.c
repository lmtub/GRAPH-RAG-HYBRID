static void qmp_input_type_int64(Visitor *v, const char *name, int64_t *obj,

                                 Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, true, errp);

    QInt *qint;



    if (!qobj) {

        return;

    }

    qint = qobject_to_qint(qobj);

    if (!qint) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "integer");

        return;

    }



    *obj = qint_get_int(qint);

}
