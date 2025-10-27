static void test_validate_fail_list(TestInputVisitorData *data,

                                     const void *unused)

{

    UserDefOneList *head = NULL;

    Error *err = NULL;

    Visitor *v;



    v = validate_test_init(data, "[ { 'string': 'string0', 'integer': 42 }, { 'string': 'string1', 'integer': 43 }, { 'string': 'string2', 'integer': 44, 'extra': 'ggg' } ]");



    visit_type_UserDefOneList(v, &head, NULL, &err);

    g_assert(err);

    error_free(err);

    qapi_free_UserDefOneList(head);

}
