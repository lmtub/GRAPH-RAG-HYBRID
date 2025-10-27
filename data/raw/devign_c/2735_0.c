static void qdict_setup(void)

{

    tests_dict = qdict_new();

    fail_unless(tests_dict != NULL);

}
