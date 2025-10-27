static void test_visitor_out_null(TestOutputVisitorData *data,

                                  const void *unused)

{

    QObject *arg;

    QDict *qdict;

    QObject *nil;



    visit_start_struct(data->ov, NULL, NULL, 0, &error_abort);

    visit_type_null(data->ov, "a", &error_abort);

    visit_check_struct(data->ov, &error_abort);

    visit_end_struct(data->ov, NULL);

    arg = visitor_get(data);

    g_assert(qobject_type(arg) == QTYPE_QDICT);

    qdict = qobject_to_qdict(arg);

    g_assert_cmpint(qdict_size(qdict), ==, 1);

    nil = qdict_get(qdict, "a");

    g_assert(nil);

    g_assert(qobject_type(nil) == QTYPE_QNULL);

}
