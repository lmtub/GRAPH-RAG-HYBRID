Visitor *qmp_output_visitor_new(QObject **result)

{

    QmpOutputVisitor *v;



    v = g_malloc0(sizeof(*v));



    v->visitor.type = VISITOR_OUTPUT;

    v->visitor.start_struct = qmp_output_start_struct;

    v->visitor.end_struct = qmp_output_end_struct;

    v->visitor.start_list = qmp_output_start_list;

    v->visitor.next_list = qmp_output_next_list;

    v->visitor.end_list = qmp_output_end_list;

    v->visitor.type_int64 = qmp_output_type_int64;

    v->visitor.type_uint64 = qmp_output_type_uint64;

    v->visitor.type_bool = qmp_output_type_bool;

    v->visitor.type_str = qmp_output_type_str;

    v->visitor.type_number = qmp_output_type_number;

    v->visitor.type_any = qmp_output_type_any;

    v->visitor.type_null = qmp_output_type_null;

    v->visitor.complete = qmp_output_complete;

    v->visitor.free = qmp_output_free;



    *result = NULL;

    v->result = result;



    return &v->visitor;

}
