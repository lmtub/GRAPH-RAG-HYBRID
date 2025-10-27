static void qmp_input_start_list(Visitor *v, const char *name, Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, true);



    if (!qobj || qobject_type(qobj) != QTYPE_QLIST) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "list");

        return;

    }



    qmp_input_push(qiv, qobj, errp);

}
