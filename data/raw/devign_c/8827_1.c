static void test_parse_path(void)

{

    g_test_trap_subprocess ("/logging/parse_path/subprocess", 0, 0);

    g_test_trap_assert_passed();

    g_test_trap_assert_stdout("");

    g_test_trap_assert_stderr("");

}
