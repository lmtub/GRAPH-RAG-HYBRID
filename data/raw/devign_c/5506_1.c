static void migrate_check_parameter(QTestState *who, const char *parameter,

                                    const char *value)

{

    QDict *rsp, *rsp_return;

    const char *result;



    rsp = wait_command(who, "{ 'execute': 'query-migrate-parameters' }");

    rsp_return = qdict_get_qdict(rsp, "return");

    result = g_strdup_printf("%" PRId64,

                             qdict_get_try_int(rsp_return,  parameter, -1));

    g_assert_cmpstr(result, ==, value);

    QDECREF(rsp);

}
