static void test_qemu_strtoul_max(void)

{

    const char *str = g_strdup_printf("%lu", ULONG_MAX);

    char f = 'X';

    const char *endptr = &f;

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, ULONG_MAX);

    g_assert(endptr == str + strlen(str));

}
