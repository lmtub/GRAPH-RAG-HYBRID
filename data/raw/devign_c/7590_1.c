Visitor *qobject_input_visitor_new_keyval(QObject *obj)

{

    QObjectInputVisitor *v = qobject_input_visitor_base_new(obj);



    v->visitor.type_int64 = qobject_input_type_int64_keyval;

    v->visitor.type_uint64 = qobject_input_type_uint64_keyval;

    v->visitor.type_bool = qobject_input_type_bool_keyval;

    v->visitor.type_str = qobject_input_type_str;

    v->visitor.type_number = qobject_input_type_number_keyval;

    v->visitor.type_any = qobject_input_type_any;

    v->visitor.type_null = qobject_input_type_null;

    v->visitor.type_size = qobject_input_type_size_keyval;



    return &v->visitor;

}
