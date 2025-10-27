static void test_validate_union_anon(TestInputVisitorData *data,

                                     const void *unused)

{

    UserDefAnonUnion *tmp = NULL;

    Visitor *v;

    Error *errp = NULL;



    v = validate_test_init(data, "42");



    visit_type_UserDefAnonUnion(v, &tmp, NULL, &errp);

    g_assert(!error_is_set(&errp));

    qapi_free_UserDefAnonUnion(tmp);

}
