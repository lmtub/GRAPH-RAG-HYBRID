static void qmp_input_type_int(Visitor *v, int64_t *obj, const char *name,

                               Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, true);



    if (!qobj || qobject_type(qobj) != QTYPE_QINT) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "integer");

        return;

    }



    *obj = qint_get_int(qobject_to_qint(qobj));

}
