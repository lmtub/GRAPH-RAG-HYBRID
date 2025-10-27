static void qmp_output_free(Visitor *v)

{

    QmpOutputVisitor *qov = to_qov(v);

    QStackEntry *e;



    while (!QSLIST_EMPTY(&qov->stack)) {

        e = QSLIST_FIRST(&qov->stack);

        QSLIST_REMOVE_HEAD(&qov->stack, node);

        g_free(e);

    }



    qobject_decref(qov->root);

    g_free(qov);

}
