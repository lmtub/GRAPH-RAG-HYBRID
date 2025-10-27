static void test_qemu_strtosz_simple(void)

{

    const char *str;

    char *endptr = NULL;

    int64_t res;



    str = "0";

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 0);

    g_assert(endptr == str + 1);



    str = "12345";

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 12345);

    g_assert(endptr == str + 5);



    res = qemu_strtosz(str, NULL);

    g_assert_cmpint(res, ==, 12345);



    /* Note: precision is 53 bits since we're parsing with strtod() */



    str = "9007199254740991"; /* 2^53-1 */

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 0x1fffffffffffff);

    g_assert(endptr == str + 16);



    str = "9007199254740992"; /* 2^53 */

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 0x20000000000000);

    g_assert(endptr == str + 16);



    str = "9007199254740993"; /* 2^53+1 */

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 0x20000000000000); /* rounded to 53 bits */

    g_assert(endptr == str + 16);



    str = "9223372036854774784"; /* 0x7ffffffffffffc00 (53 msbs set) */

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 0x7ffffffffffffc00);

    g_assert(endptr == str + 19);



    str = "9223372036854775295"; /* 0x7ffffffffffffdff */

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 0x7ffffffffffffc00); /* rounded to 53 bits */

    g_assert(endptr == str + 19);



    /* 0x7ffffffffffffe00..0x7fffffffffffffff get rounded to

     * 0x8000000000000000, thus -ERANGE; see test_qemu_strtosz_erange() */

}
