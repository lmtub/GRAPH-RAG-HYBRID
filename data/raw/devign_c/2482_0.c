static void qmp_input_free(Visitor *v)

{

    QmpInputVisitor *qiv = to_qiv(v);

    while (!QSLIST_EMPTY(&qiv->stack)) {

        StackObject *tos = QSLIST_FIRST(&qiv->stack);



        QSLIST_REMOVE_HEAD(&qiv->stack, node);

        qmp_input_stack_object_free(tos);

    }



    qobject_decref(qiv->root);

    g_free(qiv);

}
