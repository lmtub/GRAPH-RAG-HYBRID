static void qmp_input_start_alternate(Visitor *v, const char *name,

                                      GenericAlternate **obj, size_t size,

                                      bool promote_int, Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    QObject *qobj = qmp_input_get_object(qiv, name, false, errp);



    if (!qobj) {

        *obj = NULL;

        return;

    }

    *obj = g_malloc0(size);

    (*obj)->type = qobject_type(qobj);

    if (promote_int && (*obj)->type == QTYPE_QINT) {

        (*obj)->type = QTYPE_QFLOAT;

    }

}
