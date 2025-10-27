void qmp_output_visitor_cleanup(QmpOutputVisitor *v)

{

    QStackEntry *e, *tmp;



    QTAILQ_FOREACH_SAFE(e, &v->stack, node, tmp) {

        QTAILQ_REMOVE(&v->stack, e, node);

        if (e->value) {

            qobject_decref(e->value);

        }

        g_free(e);

    }



    g_free(v);

}
