START_TEST(qobject_to_qlist_test)

{

    QList *qlist;



    qlist = qlist_new();



    fail_unless(qobject_to_qlist(QOBJECT(qlist)) == qlist);



    // destroy doesn't exist yet

    g_free(qlist);

}
