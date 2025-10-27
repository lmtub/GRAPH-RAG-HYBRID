static GenericList *qmp_input_next_list(Visitor *v, GenericList **list,

                                        Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);

    GenericList *entry;

    StackObject *so = &qiv->stack[qiv->nb_stack - 1];



    if (so->entry == NULL) {

        return NULL;

    }



    entry = g_malloc0(sizeof(*entry));

    if (*list) {

        so->entry = qlist_next(so->entry);

        if (so->entry == NULL) {

            g_free(entry);

            return NULL;

        }

        (*list)->next = entry;

    }



    return entry;

}
