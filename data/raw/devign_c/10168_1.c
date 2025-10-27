static void test_qemu_strtol_full_max(void)

{

    const char *str = g_strdup_printf("%ld", LONG_MAX);

    long res;

    int err;



    err = qemu_strtol(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, LONG_MAX);

}
