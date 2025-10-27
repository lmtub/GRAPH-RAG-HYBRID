static Visitor *validate_test_init_raw(TestInputVisitorData *data,

                                       const char *json_string)

{

    return validate_test_init_internal(data, json_string, NULL);

}
