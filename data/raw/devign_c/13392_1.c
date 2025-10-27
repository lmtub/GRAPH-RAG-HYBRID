Visitor *visitor_input_test_init(TestInputVisitorData *data,

                                 const char *json_string, ...)

{

    Visitor *v;

    va_list ap;



    va_start(ap, json_string);

    data->obj = qobject_from_jsonv(json_string, &ap);

    va_end(ap);



    g_assert(data->obj != NULL);



    data->qiv = qmp_input_visitor_new(data->obj);

    g_assert(data->qiv != NULL);



    v = qmp_input_get_visitor(data->qiv);

    g_assert(v != NULL);



    return v;

}
