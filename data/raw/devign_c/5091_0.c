static void test_qemu_strtoull_max(void)

{

    char *str = g_strdup_printf("%llu", ULLONG_MAX);

    char f = 'X';

    const char *endptr = &f;

    uint64_t res = 999;

    int err;



    err = qemu_strtoull(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, ULLONG_MAX);

    g_assert(endptr == str + strlen(str));

    g_free(str);

}
