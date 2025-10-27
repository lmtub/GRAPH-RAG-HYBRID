static void test_qemu_strtoull_full_negative(void)

{

    const char *str = " \t -321";

    uint64_t res = 999;

    int err;



    err = qemu_strtoull(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 18446744073709551295LLU);

}
