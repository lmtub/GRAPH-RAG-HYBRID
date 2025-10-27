START_TEST(qdict_put_obj_test)

{

    QInt *qi;

    QDict *qdict;

    QDictEntry *ent;

    const int num = 42;



    qdict = qdict_new();



    // key "" will have tdb hash 12345

    qdict_put_obj(qdict, "", QOBJECT(qint_from_int(num)));



    fail_unless(qdict_size(qdict) == 1);

    ent = QLIST_FIRST(&qdict->table[12345 % QDICT_BUCKET_MAX]);

    qi = qobject_to_qint(ent->value);

    fail_unless(qint_get_int(qi) == num);



    // destroy doesn't exit yet

    QDECREF(qi);

    g_free(ent->key);

    g_free(ent);

    g_free(qdict);

}
