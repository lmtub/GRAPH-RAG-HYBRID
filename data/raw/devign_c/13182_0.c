static void test_qemu_strtoul_underflow(void)

{

    const char *str = "-99999999999999999999999999999999999999999999";

    char f = 'X';

    const char *endptr = &f;

    unsigned long res = 999;

    int err;



    err  = qemu_strtoul(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, -ERANGE);

    g_assert_cmpint(res, ==, -1ul);

    g_assert(endptr == str + strlen(str));

}
