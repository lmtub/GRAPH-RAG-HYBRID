START_TEST(qstring_destroy_test)

{

    QString *qstring = qstring_from_str("destroy test");

    QDECREF(qstring);

}
