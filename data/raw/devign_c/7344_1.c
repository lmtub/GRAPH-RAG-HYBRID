static void test_visitor_out_alternate(TestOutputVisitorData *data,

                                       const void *unused)

{

    QObject *arg;

    UserDefAlternate *tmp;

    QDict *qdict;



    tmp = g_new0(UserDefAlternate, 1);

    tmp->type = QTYPE_QINT;

    tmp->u.i = 42;



    visit_type_UserDefAlternate(data->ov, NULL, &tmp, &error_abort);

    arg = qmp_output_get_qobject(data->qov);



    g_assert(qobject_type(arg) == QTYPE_QINT);

    g_assert_cmpint(qint_get_int(qobject_to_qint(arg)), ==, 42);



    qapi_free_UserDefAlternate(tmp);

    qobject_decref(arg);



    tmp = g_new0(UserDefAlternate, 1);

    tmp->type = QTYPE_QSTRING;

    tmp->u.s = g_strdup("hello");



    visit_type_UserDefAlternate(data->ov, NULL, &tmp, &error_abort);

    arg = qmp_output_get_qobject(data->qov);



    g_assert(qobject_type(arg) == QTYPE_QSTRING);

    g_assert_cmpstr(qstring_get_str(qobject_to_qstring(arg)), ==, "hello");



    qapi_free_UserDefAlternate(tmp);

    qobject_decref(arg);



    tmp = g_new0(UserDefAlternate, 1);

    tmp->type = QTYPE_QDICT;

    tmp->u.udfu.integer = 1;

    tmp->u.udfu.string = g_strdup("str");

    tmp->u.udfu.enum1 = ENUM_ONE_VALUE1;

    tmp->u.udfu.u.value1 = g_new0(UserDefA, 1);

    tmp->u.udfu.u.value1->boolean = true;



    visit_type_UserDefAlternate(data->ov, NULL, &tmp, &error_abort);

    arg = qmp_output_get_qobject(data->qov);



    g_assert_cmpint(qobject_type(arg), ==, QTYPE_QDICT);

    qdict = qobject_to_qdict(arg);

    g_assert_cmpint(qdict_size(qdict), ==, 4);

    g_assert_cmpint(qdict_get_int(qdict, "integer"), ==, 1);

    g_assert_cmpstr(qdict_get_str(qdict, "string"), ==, "str");

    g_assert_cmpstr(qdict_get_str(qdict, "enum1"), ==, "value1");

    g_assert_cmpint(qdict_get_bool(qdict, "boolean"), ==, true);



    qapi_free_UserDefAlternate(tmp);

    qobject_decref(arg);

}
