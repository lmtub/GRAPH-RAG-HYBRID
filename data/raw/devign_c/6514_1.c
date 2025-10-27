static void test_qga_set_time(gconstpointer fix)

{

    const TestFixture *fixture = fix;

    QDict *ret;

    int64_t current, time;

    gchar *cmd;



    /* get current time */

    ret = qmp_fd(fixture->fd, "{'execute': 'guest-get-time'}");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    current = qdict_get_int(ret, "return");

    g_assert_cmpint(current, >, 0);

    QDECREF(ret);



    /* set some old time */

    ret = qmp_fd(fixture->fd, "{'execute': 'guest-set-time',"

                 " 'arguments': { 'time': 1000 } }");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    QDECREF(ret);



    /* check old time */

    ret = qmp_fd(fixture->fd, "{'execute': 'guest-get-time'}");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    time = qdict_get_int(ret, "return");

    g_assert_cmpint(time / 1000, <, G_USEC_PER_SEC * 10);

    QDECREF(ret);



    /* set back current time */

    cmd = g_strdup_printf("{'execute': 'guest-set-time',"

                          " 'arguments': { 'time': %" PRId64 " } }",

                          current + time * 1000);

    ret = qmp_fd(fixture->fd, cmd);

    g_free(cmd);

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    QDECREF(ret);

}
