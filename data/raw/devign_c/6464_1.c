QapiDeallocVisitor *qapi_dealloc_visitor_new(void)

{

    QapiDeallocVisitor *v;



    v = g_malloc0(sizeof(*v));



    v->visitor.start_struct = qapi_dealloc_start_struct;

    v->visitor.end_struct = qapi_dealloc_end_struct;

    v->visitor.start_implicit_struct = qapi_dealloc_start_implicit_struct;

    v->visitor.end_implicit_struct = qapi_dealloc_end_implicit_struct;

    v->visitor.start_list = qapi_dealloc_start_list;

    v->visitor.next_list = qapi_dealloc_next_list;

    v->visitor.end_list = qapi_dealloc_end_list;

    v->visitor.type_enum = qapi_dealloc_type_enum;

    v->visitor.type_int64 = qapi_dealloc_type_int64;

    v->visitor.type_uint64 = qapi_dealloc_type_uint64;

    v->visitor.type_bool = qapi_dealloc_type_bool;

    v->visitor.type_str = qapi_dealloc_type_str;

    v->visitor.type_number = qapi_dealloc_type_number;

    v->visitor.type_any = qapi_dealloc_type_anything;

    v->visitor.start_union = qapi_dealloc_start_union;



    QTAILQ_INIT(&v->stack);



    return v;

}
