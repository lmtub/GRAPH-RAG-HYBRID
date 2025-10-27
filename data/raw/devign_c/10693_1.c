QmpInputVisitor *qmp_input_visitor_new(QObject *obj, bool strict)

{

    QmpInputVisitor *v;



    v = g_malloc0(sizeof(*v));



    v->visitor.type = VISITOR_INPUT;

    v->visitor.start_struct = qmp_input_start_struct;

    v->visitor.end_struct = qmp_input_end_struct;

    v->visitor.start_list = qmp_input_start_list;

    v->visitor.next_list = qmp_input_next_list;

    v->visitor.end_list = qmp_input_end_list;

    v->visitor.start_alternate = qmp_input_start_alternate;

    v->visitor.type_int64 = qmp_input_type_int64;

    v->visitor.type_uint64 = qmp_input_type_uint64;

    v->visitor.type_bool = qmp_input_type_bool;

    v->visitor.type_str = qmp_input_type_str;

    v->visitor.type_number = qmp_input_type_number;

    v->visitor.type_any = qmp_input_type_any;

    v->visitor.type_null = qmp_input_type_null;

    v->visitor.optional = qmp_input_optional;

    v->strict = strict;



    v->root = obj;

    qobject_incref(obj);



    return v;

}
