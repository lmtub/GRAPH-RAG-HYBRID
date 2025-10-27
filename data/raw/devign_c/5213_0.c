static void test_validate_union_flat(TestInputVisitorData *data,

                                     const void *unused)

{

    UserDefFlatUnion *tmp = NULL;

    Visitor *v;

    Error *errp = NULL;



    v = validate_test_init(data,

                           "{ 'enum1': 'value1', "

                           "'string': 'str', "

                           "'boolean': true }");

    /* TODO when generator bug is fixed, add 'integer': 41 */



    visit_type_UserDefFlatUnion(v, &tmp, NULL, &errp);

    g_assert(!error_is_set(&errp));

    qapi_free_UserDefFlatUnion(tmp);

}
