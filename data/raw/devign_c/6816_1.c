static void keyword_literal(void)

{

    QObject *obj;

    QBool *qbool;

    QObject *null;

    QString *str;



    obj = qobject_from_json("true", NULL);

    qbool = qobject_to_qbool(obj);

    g_assert(qbool);

    g_assert(qbool_get_bool(qbool) == true);



    str = qobject_to_json(obj);

    g_assert(strcmp(qstring_get_str(str), "true") == 0);

    QDECREF(str);



    QDECREF(qbool);



    obj = qobject_from_json("false", NULL);

    qbool = qobject_to_qbool(obj);

    g_assert(qbool);

    g_assert(qbool_get_bool(qbool) == false);



    str = qobject_to_json(obj);

    g_assert(strcmp(qstring_get_str(str), "false") == 0);

    QDECREF(str);



    QDECREF(qbool);



    qbool = qobject_to_qbool(qobject_from_jsonf("%i", false));

    g_assert(qbool);

    g_assert(qbool_get_bool(qbool) == false);

    QDECREF(qbool);



    /* Test that non-zero values other than 1 get collapsed to true */

    qbool = qobject_to_qbool(qobject_from_jsonf("%i", 2));

    g_assert(qbool);

    g_assert(qbool_get_bool(qbool) == true);

    QDECREF(qbool);



    obj = qobject_from_json("null", NULL);

    g_assert(obj != NULL);

    g_assert(qobject_type(obj) == QTYPE_QNULL);



    null = qnull();

    g_assert(null == obj);



    qobject_decref(obj);

    qobject_decref(null);

}
