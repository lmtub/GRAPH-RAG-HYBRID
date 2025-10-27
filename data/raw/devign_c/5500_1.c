static Visitor *validate_test_init_raw(TestInputVisitorData *data,

                                       const char *json_string)

{

    Visitor *v;



    data->obj = qobject_from_json(json_string);

    g_assert(data->obj != NULL);



    data->qiv = qmp_input_visitor_new_strict(data->obj);

    g_assert(data->qiv != NULL);



    v = qmp_input_get_visitor(data->qiv);

    g_assert(v != NULL);



    return v;

}
