static void qapi_dealloc_end_list(Visitor *v)

{

    QapiDeallocVisitor *qov = to_qov(v);

    void *obj = qapi_dealloc_pop(qov);

    assert(obj == NULL); /* should've been list head tracker with no payload */

}
