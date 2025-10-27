static void test_visitor_in_uint(TestInputVisitorData *data,

                                const void *unused)

{

    Error *err = NULL;

    uint64_t res = 0;

    int64_t i64;

    double dbl;

    int value = 42;

    Visitor *v;



    v = visitor_input_test_init(data, "%d", value);



    visit_type_uint64(v, NULL, &res, &error_abort);

    g_assert_cmpuint(res, ==, (uint64_t)value);



    visit_type_int(v, NULL, &i64, &error_abort);

    g_assert_cmpint(i64, ==, value);



    visit_type_number(v, NULL, &dbl, &error_abort);

    g_assert_cmpfloat(dbl, ==, value);



    /* BUG: value between INT64_MIN and -1 accepted modulo 2^64 */

    v = visitor_input_test_init(data, "%d", -value);



    visit_type_uint64(v, NULL, &res, &error_abort);

    g_assert_cmpuint(res, ==, (uint64_t)-value);



    /* BUG: value between INT64_MAX+1 and UINT64_MAX rejected */



    v = visitor_input_test_init(data, "18446744073709551574");



    visit_type_uint64(v, NULL, &res, &err);

    error_free_or_abort(&err);



    visit_type_number(v, NULL, &dbl, &error_abort);

    g_assert_cmpfloat(dbl, ==, 18446744073709552000.0);

}
