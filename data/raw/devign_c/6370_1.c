static void qmp_input_get_next_type(Visitor *v, int *kind, const int *qobjects,

                                    const char *name, Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, false);



    if (!qobj) {

        error_setg(errp, QERR_MISSING_PARAMETER, name ? name : "null");

        return;

    }

    *kind = qobjects[qobject_type(qobj)];

}
