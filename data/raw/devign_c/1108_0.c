static void test_qemu_strtoull_overflow(void)

{

    const char *str = "99999999999999999999999999999999999999999999";

    char f = 'X';

    const char *endptr = &f;

    uint64_t res = 999;

    int err;



    err = qemu_strtoull(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, -ERANGE);

    g_assert_cmpint(res, ==, ULLONG_MAX);

    g_assert(endptr == str + strlen(str));

}
