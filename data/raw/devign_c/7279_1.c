static void test_parse_invalid_path(void)

{

    g_test_trap_subprocess ("/logging/parse_invalid_path/subprocess", 0, 0);

    g_test_trap_assert_passed();

    g_test_trap_assert_stdout("");

    g_test_trap_assert_stderr("Bad logfile format: /tmp/qemu-%d%d.log\n");

}
