static void test_native_list(TestOutputVisitorData *data,

                             const void *unused,

                             UserDefNativeListUnionKind kind)

{

    UserDefNativeListUnion *cvalue = g_new0(UserDefNativeListUnion, 1);

    QObject *obj;



    cvalue->type = kind;

    init_native_list(cvalue);



    visit_type_UserDefNativeListUnion(data->ov, NULL, &cvalue, &error_abort);



    obj = visitor_get(data);

    check_native_list(obj, cvalue->type);

    qapi_free_UserDefNativeListUnion(cvalue);

}
