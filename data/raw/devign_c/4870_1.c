static void qmp_input_type_null(Visitor *v, const char *name, Error **errp)
{
    QmpInputVisitor *qiv = to_qiv(v);
    QObject *qobj = qmp_input_get_object(qiv, name, true);
    if (qobject_type(qobj) != QTYPE_QNULL) {
        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",
                   "null");