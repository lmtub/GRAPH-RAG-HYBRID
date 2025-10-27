static void test_keyval_visit_any(void)

{

    Visitor *v;

    QDict *qdict;

    QObject *any;

    QList *qlist;

    QString *qstr;



    qdict = keyval_parse("a.0=null,a.1=1", NULL, &error_abort);

    v = qobject_input_visitor_new_keyval(QOBJECT(qdict));

    QDECREF(qdict);

    visit_start_struct(v, NULL, NULL, 0, &error_abort);

    visit_type_any(v, "a", &any, &error_abort);

    qlist = qobject_to_qlist(any);

    g_assert(qlist);

    qstr = qobject_to_qstring(qlist_pop(qlist));

    g_assert_cmpstr(qstring_get_str(qstr), ==, "null");


    qstr = qobject_to_qstring(qlist_pop(qlist));

    g_assert_cmpstr(qstring_get_str(qstr), ==, "1");

    g_assert(qlist_empty(qlist));



    visit_check_struct(v, &error_abort);

    visit_end_struct(v, NULL);

    visit_free(v);

}