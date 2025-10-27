void qtest_add_func(const char *str, void (*fn))

{

    gchar *path = g_strdup_printf("/%s/%s", qtest_get_arch(), str);

    g_test_add_func(path, fn);


}