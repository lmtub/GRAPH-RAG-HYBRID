START_TEST(qdict_destroy_simple_test)

{

    QDict *qdict;



    qdict = qdict_new();

    qdict_put_obj(qdict, "num", QOBJECT(qint_from_int(0)));

    qdict_put_obj(qdict, "str", QOBJECT(qstring_from_str("foo")));



    QDECREF(qdict);

}
