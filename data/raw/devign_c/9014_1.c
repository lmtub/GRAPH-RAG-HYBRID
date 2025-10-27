static void qmp_input_type_bool(Visitor *v, bool *obj, const char *name,

                                Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, true);



    if (!qobj || qobject_type(qobj) != QTYPE_QBOOL) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "boolean");

        return;

    }



    *obj = qbool_get_bool(qobject_to_qbool(qobj));

}
