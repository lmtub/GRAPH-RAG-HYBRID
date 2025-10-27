static void test_nop(gconstpointer data)

{

    QTestState *s;

    const char *machine = data;

    char *args;



    args = g_strdup_printf("-display none -machine %s", machine);

    s = qtest_start(args);

    if (s) {

        qtest_quit(s);

    }

    g_free(args);

}
