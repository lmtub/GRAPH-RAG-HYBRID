static void monitor_json_emitter(Monitor *mon, const QObject *data)

{

    QString *json;



    json = qobject_to_json(data);

    assert(json != NULL);



    mon->mc->print_enabled = 1;

    monitor_printf(mon, "%s\n", qstring_get_str(json));

    mon->mc->print_enabled = 0;



    QDECREF(json);

}
