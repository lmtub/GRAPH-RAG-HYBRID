static void test_qemu_strtoull_underflow(void)

{

    const char *str = "-99999999999999999999999999999999999999999999";

    char f = 'X';

    const char *endptr = &f;

    uint64_t res = 999;

    int err;



    err  = qemu_strtoull(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, -ERANGE);

    g_assert_cmpint(res, ==, -1);

    g_assert(endptr == str + strlen(str));

}
