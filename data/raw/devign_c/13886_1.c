static void test_qga_fsfreeze_and_thaw(gconstpointer fix)

{

    const TestFixture *fixture = fix;

    QDict *ret;

    const gchar *status;



    ret = qmp_fd(fixture->fd, "{'execute': 'guest-fsfreeze-freeze'}");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    QDECREF(ret);



    ret = qmp_fd(fixture->fd, "{'execute': 'guest-fsfreeze-status'}");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    status = qdict_get_try_str(ret, "return");

    g_assert_cmpstr(status, ==, "frozen");

    QDECREF(ret);



    ret = qmp_fd(fixture->fd, "{'execute': 'guest-fsfreeze-thaw'}");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    QDECREF(ret);

}
