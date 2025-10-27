StringOutputVisitor *string_output_visitor_new(bool human)

{

    StringOutputVisitor *v;



    v = g_malloc0(sizeof(*v));



    v->string = g_string_new(NULL);

    v->human = human;

    v->visitor.type = VISITOR_OUTPUT;

    v->visitor.type_int64 = print_type_int64;

    v->visitor.type_uint64 = print_type_uint64;

    v->visitor.type_size = print_type_size;

    v->visitor.type_bool = print_type_bool;

    v->visitor.type_str = print_type_str;

    v->visitor.type_number = print_type_number;

    v->visitor.start_list = start_list;

    v->visitor.next_list = next_list;

    v->visitor.end_list = end_list;

    v->visitor.free = string_output_free;



    return v;

}
