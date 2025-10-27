static void test_qemu_strtoll_full_empty(void)

{

    const char *str = "";

    int64_t res = 999;

    int err;



    err = qemu_strtoll(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 0);

}
