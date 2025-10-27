static void test_qemu_strtoul_correct(void)

{

    const char *str = "12345 foo";

    char f = 'X';

    const char *endptr = &f;

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 12345);

    g_assert(endptr == str + 5);

}
