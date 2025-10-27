static void test_qemu_strtoul_full_negative(void)

{

    const char *str = " \t -321";

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, NULL, 0, &res);

    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, -321ul);

}
