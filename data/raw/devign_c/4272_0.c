static GenericList *qmp_input_next_list(Visitor *v, GenericList **list,

                                        size_t size)

{

    QmpInputVisitor *qiv = to_qiv(v);

    GenericList *entry;

    StackObject *so = &qiv->stack[qiv->nb_stack - 1];



    if (!so->entry) {

        return NULL;

    }



    entry = g_malloc0(size);

    if (so->first) {

        *list = entry;

        so->first = false;

    } else {

        (*list)->next = entry;

    }



    return entry;

}
