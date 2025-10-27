static void test_qga_fstrim(gconstpointer fix)

{

    const TestFixture *fixture = fix;

    QDict *ret;

    QList *list;

    const QListEntry *entry;



    ret = qmp_fd(fixture->fd, "{'execute': 'guest-fstrim',"

                 " arguments: { minimum: 4194304 } }");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    list = qdict_get_qlist(ret, "return");

    entry = qlist_first(list);

    g_assert(qdict_haskey(qobject_to_qdict(entry->value), "paths"));



    QDECREF(ret);

}
