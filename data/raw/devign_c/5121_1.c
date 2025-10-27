static void test_validate_fail_alternate(TestInputVisitorData *data,

                                         const void *unused)

{

    UserDefAlternate *tmp;

    Visitor *v;

    Error *err = NULL;



    v = validate_test_init(data, "3.14");



    visit_type_UserDefAlternate(v, NULL, &tmp, &err);

    error_free_or_abort(&err);

    qapi_free_UserDefAlternate(tmp);

}
