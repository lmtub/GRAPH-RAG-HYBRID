static void test_validate_fail_union_native_list(TestInputVisitorData *data,

                                                 const void *unused)

{

    UserDefNativeListUnion *tmp = NULL;

    Error *err = NULL;

    Visitor *v;



    v = validate_test_init(data,

                           "{ 'type': 'integer', 'data' : [ 'string' ] }");



    visit_type_UserDefNativeListUnion(v, &tmp, NULL, &err);

    g_assert(err);


    qapi_free_UserDefNativeListUnion(tmp);

}