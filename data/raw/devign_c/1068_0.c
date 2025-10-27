static void bdrv_stats_iter(QObject *data, void *opaque)

{

    QDict *qdict;

    Monitor *mon = opaque;



    qdict = qobject_to_qdict(data);

    monitor_printf(mon, "%s:", qdict_get_str(qdict, "device"));



    qdict = qobject_to_qdict(qdict_get(qdict, "stats"));

    monitor_printf(mon, " rd_bytes=%" PRId64

                        " wr_bytes=%" PRId64

                        " rd_operations=%" PRId64

                        " wr_operations=%" PRId64

                        " flush_operations=%" PRId64

                        "\n",

                        qdict_get_int(qdict, "rd_bytes"),

                        qdict_get_int(qdict, "wr_bytes"),

                        qdict_get_int(qdict, "rd_operations"),

                        qdict_get_int(qdict, "wr_operations"),

                        qdict_get_int(qdict, "flush_operations"));

}
