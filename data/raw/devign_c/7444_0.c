static void test_visitor_in_any(TestInputVisitorData *data,

                                const void *unused)

{

    QObject *res = NULL;

    Error *err = NULL;

    Visitor *v;

    QInt *qint;

    QBool *qbool;

    QString *qstring;

    QDict *qdict;

    QObject *qobj;



    v = visitor_input_test_init(data, "-42");

    visit_type_any(v, &res, NULL, &err);

    g_assert(!err);

    qint = qobject_to_qint(res);

    g_assert(qint);

    g_assert_cmpint(qint_get_int(qint), ==, -42);

    qobject_decref(res);



    v = visitor_input_test_init(data, "{ 'integer': -42, 'boolean': true, 'string': 'foo' }");

    visit_type_any(v, &res, NULL, &err);

    g_assert(!err);

    qdict = qobject_to_qdict(res);

    g_assert(qdict && qdict_size(qdict) == 3);

    qobj = qdict_get(qdict, "integer");

    g_assert(qobj);

    qint = qobject_to_qint(qobj);

    g_assert(qint);

    g_assert_cmpint(qint_get_int(qint), ==, -42);

    qobj = qdict_get(qdict, "boolean");

    g_assert(qobj);

    qbool = qobject_to_qbool(qobj);

    g_assert(qbool);

    g_assert(qbool_get_bool(qbool) == true);

    qobj = qdict_get(qdict, "string");

    g_assert(qobj);

    qstring = qobject_to_qstring(qobj);

    g_assert(qstring);

    g_assert_cmpstr(qstring_get_str(qstring), ==, "foo");

    qobject_decref(res);

}
