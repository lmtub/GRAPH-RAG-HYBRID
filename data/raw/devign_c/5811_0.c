static GenericList *qmp_output_next_list(Visitor *v, GenericList **listp,

                                         size_t size)

{

    GenericList *list = *listp;

    QmpOutputVisitor *qov = to_qov(v);

    QStackEntry *e = QTAILQ_FIRST(&qov->stack);



    assert(e);

    if (e->is_list_head) {

        e->is_list_head = false;

        return list;

    }



    return list ? list->next : NULL;

}
