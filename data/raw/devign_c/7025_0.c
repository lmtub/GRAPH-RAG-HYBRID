START_TEST(qobject_to_qdict_test)

{

    fail_unless(qobject_to_qdict(QOBJECT(tests_dict)) == tests_dict);

}
