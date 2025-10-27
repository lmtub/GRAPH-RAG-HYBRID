static void test_qemu_strtoll_whitespace(void)

{

    const char *str = "  \t  ";

    char f = 'X';

    const char *endptr = &f;

    int64_t res = 999;

    int err;



    err = qemu_strtoll(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 0);

    g_assert(endptr == str);

}
