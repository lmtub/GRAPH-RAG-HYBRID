static GenericList *qobject_input_next_list(Visitor *v, GenericList *tail,

                                            size_t size)

{

    QObjectInputVisitor *qiv = to_qiv(v);

    StackObject *so = QSLIST_FIRST(&qiv->stack);



    if (!so->entry) {

        return NULL;

    }

    tail->next = g_malloc0(size);

    return tail->next;

}
