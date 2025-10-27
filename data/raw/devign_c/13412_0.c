static void test_visitor_out_bool(TestOutputVisitorData *data,

                                  const void *unused)

{

    bool value = true;

    QObject *obj;



    visit_type_bool(data->ov, NULL, &value, &error_abort);



    obj = visitor_get(data);

    g_assert(qobject_type(obj) == QTYPE_QBOOL);

    g_assert(qbool_get_bool(qobject_to_qbool(obj)) == value);

}
