static void test_visitor_in_number(TestInputVisitorData *data,

                                   const void *unused)

{

    double res = 0, value = 3.14;

    Visitor *v;



    v = visitor_input_test_init(data, "%f", value);



    visit_type_number(v, NULL, &res, &error_abort);

    g_assert_cmpfloat(res, ==, value);

}
