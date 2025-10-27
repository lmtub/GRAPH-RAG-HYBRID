START_TEST(qstring_from_substr_test)

{

    QString *qs;



    qs = qstring_from_substr("virtualization", 3, 9);

    fail_unless(qs != NULL);

    fail_unless(strcmp(qstring_get_str(qs), "tualiza") == 0);



    QDECREF(qs);

}
