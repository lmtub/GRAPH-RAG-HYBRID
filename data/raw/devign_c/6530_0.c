static void validate_test_add(const char *testpath,

                               TestInputVisitorData *data,

                               void (*test_func)(TestInputVisitorData *data, const void *user_data))

{

    g_test_add(testpath, TestInputVisitorData, data, NULL, test_func,

               validate_teardown);

}
