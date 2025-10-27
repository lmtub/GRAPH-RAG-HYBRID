static void test_validate_union(TestInputVisitorData *data,

                                 const void *unused)

{

    UserDefUnion *tmp = NULL;

    Visitor *v;

    Error *err = NULL;



    v = validate_test_init(data, "{ 'type': 'b', 'integer': 41, 'data' : { 'integer': 42 } }");



    visit_type_UserDefUnion(v, &tmp, NULL, &err);

    g_assert(!err);

    qapi_free_UserDefUnion(tmp);

}
