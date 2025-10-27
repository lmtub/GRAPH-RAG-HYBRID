static void qmp_output_push_obj(QmpOutputVisitor *qov, QObject *value)

{

    QStackEntry *e = g_malloc0(sizeof(*e));



    assert(qov->root);

    assert(value);

    e->value = value;

    if (qobject_type(e->value) == QTYPE_QLIST) {

        e->is_list_head = true;

    }

    QTAILQ_INSERT_HEAD(&qov->stack, e, node);

}
