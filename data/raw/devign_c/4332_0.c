QapiDeallocVisitor *qapi_dealloc_visitor_new(void)

{

    QapiDeallocVisitor *v;



    v = g_malloc0(sizeof(*v));



    v->visitor.type = VISITOR_DEALLOC;

    v->visitor.start_struct = qapi_dealloc_start_struct;

    v->visitor.end_struct = qapi_dealloc_end_struct;

    v->visitor.start_alternate = qapi_dealloc_start_alternate;

    v->visitor.end_alternate = qapi_dealloc_end_alternate;

    v->visitor.start_list = qapi_dealloc_start_list;

    v->visitor.next_list = qapi_dealloc_next_list;

    v->visitor.end_list = qapi_dealloc_end_list;

    v->visitor.type_int64 = qapi_dealloc_type_int64;

    v->visitor.type_uint64 = qapi_dealloc_type_uint64;

    v->visitor.type_bool = qapi_dealloc_type_bool;

    v->visitor.type_str = qapi_dealloc_type_str;

    v->visitor.type_number = qapi_dealloc_type_number;

    v->visitor.type_any = qapi_dealloc_type_anything;

    v->visitor.type_null = qapi_dealloc_type_null;



    return v;

}
