Visitor *visitor_input_test_init(TestInputVisitorData *data,

                                 const char *json_string, ...)

{

    Visitor *v;

    va_list ap;



    va_start(ap, json_string);

    v = visitor_input_test_init_internal(data, json_string, &ap);

    va_end(ap);

    return v;

}
