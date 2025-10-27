static void test_qemu_strtoull_full_correct(void)

{

    const char *str = "18446744073709551614";

    uint64_t res = 999;

    int err;



    err = qemu_strtoull(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 18446744073709551614LLU);

}
