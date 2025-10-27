static void test_qemu_strtoul_full_max(void)

{

    const char *str = g_strdup_printf("%lu", ULONG_MAX);

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, ULONG_MAX);

}
