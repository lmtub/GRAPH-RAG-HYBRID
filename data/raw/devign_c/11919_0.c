static void test_qemu_strtoul_decimal(void)

{

    const char *str = "0123";

    char f = 'X';

    const char *endptr = &f;

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, &endptr, 10, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 123);

    g_assert(endptr == str + strlen(str));



    str = "123";

    res = 999;

    endptr = &f;

    err = qemu_strtoul(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 123);

    g_assert(endptr == str + strlen(str));

}
