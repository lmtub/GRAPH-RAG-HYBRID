static void qmp_input_type_number(Visitor *v, double *obj, const char *name,

                                  Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, true);



    if (!qobj || (qobject_type(qobj) != QTYPE_QFLOAT &&

        qobject_type(qobj) != QTYPE_QINT)) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "number");

        return;

    }



    if (qobject_type(qobj) == QTYPE_QINT) {

        *obj = qint_get_int(qobject_to_qint(qobj));

    } else {

        *obj = qfloat_get_double(qobject_to_qfloat(qobj));

    }

}
