static void test_validate_list(TestInputVisitorData *data,

                                const void *unused)

{

    UserDefOneList *head = NULL;

    Visitor *v;



    v = validate_test_init(data, "[ { 'string': 'string0', 'integer': 42 }, { 'string': 'string1', 'integer': 43 }, { 'string': 'string2', 'integer': 44 } ]");



    visit_type_UserDefOneList(v, NULL, &head, &error_abort);

    qapi_free_UserDefOneList(head);

}
