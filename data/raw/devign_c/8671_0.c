START_TEST(qdict_get_not_exists_test)

{

    fail_unless(qdict_get(tests_dict, "foo") == NULL);

}
