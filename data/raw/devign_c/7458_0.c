static void migrate_set_downtime(QTestState *who, const char *value)

{

    QDict *rsp;

    gchar *cmd;



    cmd = g_strdup_printf("{ 'execute': 'migrate_set_downtime',"

                          "'arguments': { 'value': %s } }", value);

    rsp = qtest_qmp(who, cmd);

    g_free(cmd);

    g_assert(qdict_haskey(rsp, "return"));

    QDECREF(rsp);

}
