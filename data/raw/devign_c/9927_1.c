static void test_keyval_visit_alternate(void)

{

    Error *err = NULL;

    Visitor *v;

    QDict *qdict;

    AltNumStr *ans;

    AltNumInt *ani;



    /*

     * Can't do scalar alternate variants other than string.  You get

     * the string variant if there is one, else an error.

     */

    qdict = keyval_parse("a=1,b=2", NULL, &error_abort);

    v = qobject_input_visitor_new_keyval(QOBJECT(qdict));

    QDECREF(qdict);

    visit_start_struct(v, NULL, NULL, 0, &error_abort);

    visit_type_AltNumStr(v, "a", &ans, &error_abort);

    g_assert_cmpint(ans->type, ==, QTYPE_QSTRING);

    g_assert_cmpstr(ans->u.s, ==, "1");

    qapi_free_AltNumStr(ans);

    visit_type_AltNumInt(v, "a", &ani, &err);

    error_free_or_abort(&err);

    visit_end_struct(v, NULL);

    visit_free(v);

}
