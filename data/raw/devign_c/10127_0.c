static void test_qemu_strtoul_trailing(void)

{

    const char *str = "123xxx";

    char f = 'X';

    const char *endptr = &f;

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 123);

    g_assert(endptr == str + 3);

}
