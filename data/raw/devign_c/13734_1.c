static void test_validate_fail_struct_nested(TestInputVisitorData *data,

                                              const void *unused)

{

    UserDefTwo *udp = NULL;

    Error *err = NULL;

    Visitor *v;



    v = validate_test_init(data, "{ 'string0': 'string0', 'dict1': { 'string1': 'string1', 'dict2': { 'userdef1': { 'integer': 42, 'string': 'string', 'extra': [42, 23, {'foo':'bar'}] }, 'string2': 'string2'}}}");



    visit_type_UserDefTwo(v, NULL, &udp, &err);

    error_free_or_abort(&err);

    qapi_free_UserDefTwo(udp);

}
