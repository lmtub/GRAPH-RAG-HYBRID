START_TEST(qobject_to_qint_test)

{

    QInt *qi;



    qi = qint_from_int(0);

    fail_unless(qobject_to_qint(QOBJECT(qi)) == qi);



    QDECREF(qi);

}
