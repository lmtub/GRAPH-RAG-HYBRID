START_TEST(qdict_get_int_test)

{

    int ret;

    const int value = 100;

    const char *key = "int";



    qdict_put(tests_dict, key, qint_from_int(value));



    ret = qdict_get_int(tests_dict, key);

    fail_unless(ret == value);

}
