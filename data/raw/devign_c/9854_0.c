static void test_qemu_strtosz_float(void)

{

    const char *str = "12.345M";

    char *endptr = NULL;

    int64_t res;



    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 12.345 * M_BYTE);

    g_assert(endptr == str + 7);

}
