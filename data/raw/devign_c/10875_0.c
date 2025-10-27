static void test_validate_fail_union_flat(TestInputVisitorData *data,

                                          const void *unused)

{

    UserDefFlatUnion *tmp = NULL;

    Error *errp = NULL;

    Visitor *v;



    v = validate_test_init(data, "{ 'string': 'c', 'integer': 41, 'boolean': true }");



    visit_type_UserDefFlatUnion(v, &tmp, NULL, &errp);

    g_assert(error_is_set(&errp));

    qapi_free_UserDefFlatUnion(tmp);

}
