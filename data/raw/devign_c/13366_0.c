static void monitor_protocol_emitter(Monitor *mon, QObject *data)

{

    QDict *qmp;



    qmp = qdict_new();



    if (!monitor_has_error(mon)) {

        /* success response */

        if (data) {

            assert(qobject_type(data) == QTYPE_QDICT);

            qobject_incref(data);

            qdict_put_obj(qmp, "return", data);

        } else {

            /* return an empty QDict by default */

            qdict_put(qmp, "return", qdict_new());

        }

    } else {

        /* error response */

        qdict_put(mon->error->error, "desc", qerror_human(mon->error));

        qdict_put(qmp, "error", mon->error->error);

        QINCREF(mon->error->error);

        QDECREF(mon->error);

        mon->error = NULL;

    }



    if (mon->mc->id) {

        qdict_put_obj(qmp, "id", mon->mc->id);

        mon->mc->id = NULL;

    }



    monitor_json_emitter(mon, QOBJECT(qmp));

    QDECREF(qmp);

}
