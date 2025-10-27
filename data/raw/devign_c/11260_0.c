StringInputVisitor *string_input_visitor_new(const char *str)

{

    StringInputVisitor *v;



    v = g_malloc0(sizeof(*v));



    v->visitor.type = VISITOR_INPUT;

    v->visitor.type_int64 = parse_type_int64;

    v->visitor.type_uint64 = parse_type_uint64;

    v->visitor.type_size = parse_type_size;

    v->visitor.type_bool = parse_type_bool;

    v->visitor.type_str = parse_type_str;

    v->visitor.type_number = parse_type_number;

    v->visitor.start_list = start_list;

    v->visitor.next_list = next_list;

    v->visitor.end_list = end_list;

    v->visitor.optional = parse_optional;



    v->string = str;

    v->head = true;

    return v;

}
