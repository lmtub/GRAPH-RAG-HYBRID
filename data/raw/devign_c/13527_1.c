static void test_visitor_in_alternate(TestInputVisitorData *data,

                                      const void *unused)

{

    Visitor *v;

    Error *err = NULL;

    UserDefAlternate *tmp;



    v = visitor_input_test_init(data, "42");

    visit_type_UserDefAlternate(v, &tmp, NULL, &error_abort);

    g_assert_cmpint(tmp->type, ==, USER_DEF_ALTERNATE_KIND_I);

    g_assert_cmpint(tmp->u.i, ==, 42);

    qapi_free_UserDefAlternate(tmp);



    v = visitor_input_test_init(data, "'string'");

    visit_type_UserDefAlternate(v, &tmp, NULL, &error_abort);

    g_assert_cmpint(tmp->type, ==, USER_DEF_ALTERNATE_KIND_S);

    g_assert_cmpstr(tmp->u.s, ==, "string");

    qapi_free_UserDefAlternate(tmp);



    v = visitor_input_test_init(data, "false");

    visit_type_UserDefAlternate(v, &tmp, NULL, &err);

    g_assert(err);

    error_free(err);

    err = NULL;

    qapi_free_UserDefAlternate(tmp);

}
