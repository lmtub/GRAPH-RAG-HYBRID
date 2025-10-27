static void test_dealloc_partial(void)

{

    static const char text[] = "don't leak me";



    UserDefTwo *ud2 = NULL;

    Error *err = NULL;



    /* create partial object */

    {

        QDict *ud2_dict;

        QmpInputVisitor *qiv;



        ud2_dict = qdict_new();

        qdict_put_obj(ud2_dict, "string0", QOBJECT(qstring_from_str(text)));



        qiv = qmp_input_visitor_new(QOBJECT(ud2_dict), false);

        visit_type_UserDefTwo(qmp_input_get_visitor(qiv), NULL, &ud2, &err);

        qmp_input_visitor_cleanup(qiv);

        QDECREF(ud2_dict);

    }



    /* verify partial success */

    assert(ud2 != NULL);

    assert(ud2->string0 != NULL);

    assert(strcmp(ud2->string0, text) == 0);

    assert(ud2->dict1 == NULL);



    /* confirm & release construction error */

    error_free_or_abort(&err);



    /* tear down partial object */

    qapi_free_UserDefTwo(ud2);

}
