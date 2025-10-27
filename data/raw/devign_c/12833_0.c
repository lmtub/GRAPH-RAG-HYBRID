static void test_visitor_in_struct_nested(TestInputVisitorData *data,

                                          const void *unused)

{

    UserDefTwo *udp = NULL;

    Visitor *v;



    v = visitor_input_test_init(data, "{ 'string0': 'string0', "

                                "'dict1': { 'string1': 'string1', "

                                "'dict2': { 'userdef': { 'integer': 42, "

                                "'string': 'string' }, 'string': 'string2'}}}");



    visit_type_UserDefTwo(v, NULL, &udp, &error_abort);



    g_assert_cmpstr(udp->string0, ==, "string0");

    g_assert_cmpstr(udp->dict1->string1, ==, "string1");

    g_assert_cmpint(udp->dict1->dict2->userdef->integer, ==, 42);

    g_assert_cmpstr(udp->dict1->dict2->userdef->string, ==, "string");

    g_assert_cmpstr(udp->dict1->dict2->string, ==, "string2");

    g_assert(udp->dict1->has_dict3 == false);



    qapi_free_UserDefTwo(udp);

}
