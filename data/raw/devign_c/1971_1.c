static void monitor_qmp_event(void *opaque, int event)

{

    QObject *data;

    Monitor *mon = opaque;



    switch (event) {

    case CHR_EVENT_OPENED:

        mon->qmp.in_command_mode = false;

        data = get_qmp_greeting();

        monitor_json_emitter(mon, data);

        qobject_decref(data);

        mon_refcount++;

        break;

    case CHR_EVENT_CLOSED:

        json_message_parser_destroy(&mon->qmp.parser);

        json_message_parser_init(&mon->qmp.parser, handle_qmp_command);

        mon_refcount--;

        monitor_fdsets_cleanup();

        break;

    }

}
