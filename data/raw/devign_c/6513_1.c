static void add_pc_test_cases(void)

{

    QDict *response, *minfo;

    QList *list;

    const QListEntry *p;

    QObject *qobj;

    QString *qstr;

    const char *mname, *path;

    PCTestData *data;



    qtest_start("-machine none");

    response = qmp("{ 'execute': 'query-machines' }");

    g_assert(response);

    list = qdict_get_qlist(response, "return");

    g_assert(list);



    for (p = qlist_first(list); p; p = qlist_next(p)) {

        minfo = qobject_to_qdict(qlist_entry_obj(p));

        g_assert(minfo);

        qobj = qdict_get(minfo, "name");

        g_assert(qobj);

        qstr = qobject_to_qstring(qobj);

        g_assert(qstr);

        mname = qstring_get_str(qstr);

        if (!g_str_has_prefix(mname, "pc-")) {

            continue;

        }

        data = g_malloc(sizeof(PCTestData));

        data->machine = mname;

        data->cpu_model = "Haswell"; /* 1.3+ theoretically */

        data->sockets = 1;

        data->cores = 3;

        data->threads = 2;

        data->maxcpus = data->sockets * data->cores * data->threads * 2;

        if (g_str_has_suffix(mname, "-1.4") ||

            (strcmp(mname, "pc-1.3") == 0) ||

            (strcmp(mname, "pc-1.2") == 0) ||

            (strcmp(mname, "pc-1.1") == 0) ||

            (strcmp(mname, "pc-1.0") == 0) ||

            (strcmp(mname, "pc-0.15") == 0) ||

            (strcmp(mname, "pc-0.14") == 0) ||

            (strcmp(mname, "pc-0.13") == 0) ||

            (strcmp(mname, "pc-0.12") == 0) ||

            (strcmp(mname, "pc-0.11") == 0) ||

            (strcmp(mname, "pc-0.10") == 0)) {

            path = g_strdup_printf("cpu/%s/init/%ux%ux%u&maxcpus=%u",

                                   mname, data->sockets, data->cores,

                                   data->threads, data->maxcpus);

            qtest_add_data_func(path, data, test_pc_without_cpu_add);

        } else {

            path = g_strdup_printf("cpu/%s/add/%ux%ux%u&maxcpus=%u",

                                   mname, data->sockets, data->cores,

                                   data->threads, data->maxcpus);

            qtest_add_data_func(path, data, test_pc_with_cpu_add);

        }

    }

    qtest_end();

}
