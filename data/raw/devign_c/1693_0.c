static void test_visitor_in_alternate_number(TestInputVisitorData *data,

                                             const void *unused)

{

    Visitor *v;

    Error *err = NULL;

    AltStrBool *asb;

    AltStrNum *asn;

    AltNumStr *ans;

    AltStrInt *asi;

    AltIntNum *ain;

    AltNumInt *ani;



    /* Parsing an int */



    v = visitor_input_test_init(data, "42");

    visit_type_AltStrBool(v, &asb, NULL, &err);

    g_assert(err);

    error_free(err);

    err = NULL;

    qapi_free_AltStrBool(asb);

    visitor_input_teardown(data, NULL);



    /* FIXME: Order of alternate should not affect semantics; asn should

     * parse the same as ans */

    v = visitor_input_test_init(data, "42");

    visit_type_AltStrNum(v, &asn, NULL, &err);

    /* FIXME g_assert_cmpint(asn->kind, == ALT_STR_NUM_KIND_N); */

    /* FIXME g_assert_cmpfloat(asn->n, ==, 42); */

    g_assert(err);

    error_free(err);

    err = NULL;

    qapi_free_AltStrNum(asn);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42");

    visit_type_AltNumStr(v, &ans, NULL, &error_abort);

    g_assert_cmpint(ans->kind, ==, ALT_NUM_STR_KIND_N);

    g_assert_cmpfloat(ans->n, ==, 42);

    qapi_free_AltNumStr(ans);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42");

    visit_type_AltStrInt(v, &asi, NULL, &error_abort);

    g_assert_cmpint(asi->kind, ==, ALT_STR_INT_KIND_I);

    g_assert_cmpint(asi->i, ==, 42);

    qapi_free_AltStrInt(asi);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42");

    visit_type_AltIntNum(v, &ain, NULL, &error_abort);

    g_assert_cmpint(ain->kind, ==, ALT_INT_NUM_KIND_I);

    g_assert_cmpint(ain->i, ==, 42);

    qapi_free_AltIntNum(ain);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42");

    visit_type_AltNumInt(v, &ani, NULL, &error_abort);

    g_assert_cmpint(ani->kind, ==, ALT_NUM_INT_KIND_I);

    g_assert_cmpint(ani->i, ==, 42);

    qapi_free_AltNumInt(ani);

    visitor_input_teardown(data, NULL);



    /* Parsing a double */



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltStrBool(v, &asb, NULL, &err);

    g_assert(err);

    error_free(err);

    err = NULL;

    qapi_free_AltStrBool(asb);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltStrNum(v, &asn, NULL, &error_abort);

    g_assert_cmpint(asn->kind, ==, ALT_STR_NUM_KIND_N);

    g_assert_cmpfloat(asn->n, ==, 42.5);

    qapi_free_AltStrNum(asn);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltNumStr(v, &ans, NULL, &error_abort);

    g_assert_cmpint(ans->kind, ==, ALT_NUM_STR_KIND_N);

    g_assert_cmpfloat(ans->n, ==, 42.5);

    qapi_free_AltNumStr(ans);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltStrInt(v, &asi, NULL, &err);

    g_assert(err);

    error_free(err);

    err = NULL;

    qapi_free_AltStrInt(asi);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltIntNum(v, &ain, NULL, &error_abort);

    g_assert_cmpint(ain->kind, ==, ALT_INT_NUM_KIND_N);

    g_assert_cmpfloat(ain->n, ==, 42.5);

    qapi_free_AltIntNum(ain);

    visitor_input_teardown(data, NULL);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltNumInt(v, &ani, NULL, &error_abort);

    g_assert_cmpint(ani->kind, ==, ALT_NUM_INT_KIND_N);

    g_assert_cmpfloat(ani->n, ==, 42.5);

    qapi_free_AltNumInt(ani);

    visitor_input_teardown(data, NULL);

}
