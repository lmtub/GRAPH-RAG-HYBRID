static void test_validate_alternate(TestInputVisitorData *data,

                                    const void *unused)

{

    UserDefAlternate *tmp = NULL;

    Visitor *v;



    v = validate_test_init(data, "42");



    visit_type_UserDefAlternate(v, NULL, &tmp, &error_abort);

    qapi_free_UserDefAlternate(tmp);

}
