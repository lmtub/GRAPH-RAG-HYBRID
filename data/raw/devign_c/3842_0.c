START_TEST(qdict_get_str_test)

{

    const char *p;

    const char *key = "key";

    const char *str = "string";



    qdict_put(tests_dict, key, qstring_from_str(str));



    p = qdict_get_str(tests_dict, key);

    fail_unless(p != NULL);

    fail_unless(strcmp(p, str) == 0);

}
