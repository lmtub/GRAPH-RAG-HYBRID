static GenericList *qapi_dealloc_next_list(Visitor *v, GenericList **listp,

                                           size_t size)

{

    GenericList *list = *listp;

    QapiDeallocVisitor *qov = to_qov(v);

    StackEntry *e = QTAILQ_FIRST(&qov->stack);



    if (e && e->is_list_head) {

        e->is_list_head = false;

        return list;

    }



    if (list) {

        list = list->next;

        g_free(*listp);

        return list;

    }



    return NULL;

}
