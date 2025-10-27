START_TEST(keyword_literal)

{

    QObject *obj;

    QBool *qbool;

    QString *str;



    obj = qobject_from_json("true");

    fail_unless(obj != NULL);

    fail_unless(qobject_type(obj) == QTYPE_QBOOL);



    qbool = qobject_to_qbool(obj);

    fail_unless(qbool_get_int(qbool) != 0);



    str = qobject_to_json(obj);

    fail_unless(strcmp(qstring_get_str(str), "true") == 0);

    QDECREF(str);



    QDECREF(qbool);



    obj = qobject_from_json("false");

    fail_unless(obj != NULL);

    fail_unless(qobject_type(obj) == QTYPE_QBOOL);



    qbool = qobject_to_qbool(obj);

    fail_unless(qbool_get_int(qbool) == 0);



    str = qobject_to_json(obj);

    fail_unless(strcmp(qstring_get_str(str), "false") == 0);

    QDECREF(str);



    QDECREF(qbool);



    obj = qobject_from_jsonf("%i", false);

    fail_unless(obj != NULL);

    fail_unless(qobject_type(obj) == QTYPE_QBOOL);



    qbool = qobject_to_qbool(obj);

    fail_unless(qbool_get_int(qbool) == 0);



    QDECREF(qbool);

    

    obj = qobject_from_jsonf("%i", true);

    fail_unless(obj != NULL);

    fail_unless(qobject_type(obj) == QTYPE_QBOOL);



    qbool = qobject_to_qbool(obj);

    fail_unless(qbool_get_int(qbool) != 0);



    QDECREF(qbool);

}
