static void test_visitor_in_int(TestInputVisitorData *data,

                                const void *unused)

{

    int64_t res = 0, value = -42;

    Visitor *v;



    v = visitor_input_test_init(data, "%" PRId64, value);



    visit_type_int(v, NULL, &res, &error_abort);

    g_assert_cmpint(res, ==, value);

}
