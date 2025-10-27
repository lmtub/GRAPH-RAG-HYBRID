static void qapi_dealloc_end_struct(Visitor *v, Error **errp)

{

    QapiDeallocVisitor *qov = to_qov(v);

    void **obj = qapi_dealloc_pop(qov);

    if (obj) {

        g_free(*obj);

    }

}
