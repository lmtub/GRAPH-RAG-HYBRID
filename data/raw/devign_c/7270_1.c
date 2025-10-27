static void test_qemu_strtol_max(void)

{

    const char *str = g_strdup_printf("%ld", LONG_MAX);

    char f = 'X';

    const char *endptr = &f;

    long res = 999;

    int err;



    err = qemu_strtol(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, LONG_MAX);

    g_assert(endptr == str + strlen(str));

}
