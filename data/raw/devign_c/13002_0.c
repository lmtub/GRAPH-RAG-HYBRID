static void test_visitor_out_string(TestOutputVisitorData *data,

                                    const void *unused)

{

    char *string = (char *) "Q E M U";

    Error *err = NULL;

    QObject *obj;



    visit_type_str(data->ov, &string, NULL, &err);

    g_assert(!err);



    obj = qmp_output_get_qobject(data->qov);

    g_assert(obj != NULL);

    g_assert(qobject_type(obj) == QTYPE_QSTRING);

    g_assert_cmpstr(qstring_get_str(qobject_to_qstring(obj)), ==, string);



    qobject_decref(obj);

}
