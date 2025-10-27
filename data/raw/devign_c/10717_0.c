static void test_validate_union_native_list(TestInputVisitorData *data,

                                            const void *unused)

{

    UserDefNativeListUnion *tmp = NULL;

    Visitor *v;

    Error *err = NULL;



    v = validate_test_init(data, "{ 'type': 'integer', 'data' : [ 1, 2 ] }");



    visit_type_UserDefNativeListUnion(v, &tmp, NULL, &err);

    g_assert(!err);

    qapi_free_UserDefNativeListUnion(tmp);

}
