static void info_mice_iter(QObject *data, void *opaque)

{

    QDict *mouse;

    Monitor *mon = opaque;



    mouse = qobject_to_qdict(data);

    monitor_printf(mon, "%c Mouse #%" PRId64 ": %s\n",

                  (qdict_get_bool(mouse, "current") ? '*' : ' '),

                  qdict_get_int(mouse, "index"), qdict_get_str(mouse, "name"));

}
