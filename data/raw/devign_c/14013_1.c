static void test_visitor_in_union_flat(TestInputVisitorData *data,

                                       const void *unused)

{

    Visitor *v;

    UserDefFlatUnion *tmp;

    UserDefUnionBase *base;



    v = visitor_input_test_init(data,

                                "{ 'enum1': 'value1', "

                                "'integer': 41, "

                                "'string': 'str', "

                                "'boolean': true }");



    visit_type_UserDefFlatUnion(v, NULL, &tmp, &error_abort);

    g_assert_cmpint(tmp->enum1, ==, ENUM_ONE_VALUE1);

    g_assert_cmpstr(tmp->string, ==, "str");

    g_assert_cmpint(tmp->integer, ==, 41);

    g_assert_cmpint(tmp->u.value1->boolean, ==, true);



    base = qapi_UserDefFlatUnion_base(tmp);

    g_assert(&base->enum1 == &tmp->enum1);



    qapi_free_UserDefFlatUnion(tmp);

}
