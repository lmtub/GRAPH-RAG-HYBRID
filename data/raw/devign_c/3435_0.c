START_TEST(qdict_put_exists_test)

{

    int value;

    const char *key = "exists";



    qdict_put(tests_dict, key, qint_from_int(1));

    qdict_put(tests_dict, key, qint_from_int(2));



    value = qdict_get_int(tests_dict, key);

    fail_unless(value == 2);



    fail_unless(qdict_size(tests_dict) == 1);

}
