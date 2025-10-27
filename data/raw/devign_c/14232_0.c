START_TEST(qstring_get_str_test)

{

    QString *qstring;

    const char *ret_str;

    const char *str = "QEMU/KVM";



    qstring = qstring_from_str(str);

    ret_str = qstring_get_str(qstring);

    fail_unless(strcmp(ret_str, str) == 0);



    QDECREF(qstring);

}
