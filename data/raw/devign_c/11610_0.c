int select_watchdog_action(const char *p)

{

    int action;

    char *qapi_value;



    qapi_value = g_ascii_strdown(p, -1);

    action = qapi_enum_parse(&WatchdogAction_lookup, qapi_value, -1, NULL);

    g_free(qapi_value);

    if (action < 0)

        return -1;

    watchdog_action = action;

    return 0;

}
