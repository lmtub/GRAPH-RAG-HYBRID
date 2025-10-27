static void test_qemu_strtoul_octal(void)

{

    const char *str = "0123";

    char f = 'X';

    const char *endptr = &f;

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, &endptr, 8, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 0123);

    g_assert(endptr == str + strlen(str));



    res = 999;

    endptr = &f;

    err = qemu_strtoul(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 0123);

    g_assert(endptr == str + strlen(str));

}
