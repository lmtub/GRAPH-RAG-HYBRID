START_TEST(qint_get_int_test)

{

    QInt *qi;

    const int value = 123456;



    qi = qint_from_int(value);

    fail_unless(qint_get_int(qi) == value);



    QDECREF(qi);

}
