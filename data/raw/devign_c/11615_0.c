static void test_visitor_out_struct(TestOutputVisitorData *data,

                                    const void *unused)

{

    TestStruct test_struct = { .integer = 42,

                               .boolean = false,

                               .string = (char *) "foo"};

    TestStruct *p = &test_struct;

    QObject *obj;

    QDict *qdict;



    visit_type_TestStruct(data->ov, NULL, &p, &error_abort);



    obj = visitor_get(data);

    g_assert(qobject_type(obj) == QTYPE_QDICT);



    qdict = qobject_to_qdict(obj);

    g_assert_cmpint(qdict_size(qdict), ==, 3);

    g_assert_cmpint(qdict_get_int(qdict, "integer"), ==, 42);

    g_assert_cmpint(qdict_get_bool(qdict, "boolean"), ==, false);

    g_assert_cmpstr(qdict_get_str(qdict, "string"), ==, "foo");

}
