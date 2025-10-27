static void test_validate_fail_union_flat(TestInputVisitorData *data,

                                          const void *unused)

{

    UserDefFlatUnion *tmp = NULL;

    Error *err = NULL;

    Visitor *v;



    v = validate_test_init(data, "{ 'string': 'c', 'integer': 41, 'boolean': true }");



    visit_type_UserDefFlatUnion(v, &tmp, NULL, &err);

    g_assert(err);


    qapi_free_UserDefFlatUnion(tmp);

}