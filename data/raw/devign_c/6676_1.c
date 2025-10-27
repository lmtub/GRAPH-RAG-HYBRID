static QList *get_cpus(QDict **resp)

{

    *resp = qmp("{ 'execute': 'query-cpus' }");

    g_assert(*resp);

    g_assert(qdict_haskey(*resp, "return"));

    return  qdict_get_qlist(*resp, "return");

}
