static void qmp_input_optional(Visitor *v, const char *name, bool *present)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, false, NULL);



    if (!qobj) {

        *present = false;

        return;

    }



    *present = true;

}
