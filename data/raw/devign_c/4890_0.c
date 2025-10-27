static void monitor_control_event(void *opaque, int event)

{

    if (event == CHR_EVENT_OPENED) {

        QObject *data;

        Monitor *mon = opaque;



        json_message_parser_init(&mon->mc->parser, handle_qmp_command);



        data = qobject_from_jsonf("{ 'QMP': { 'capabilities': [] } }");

        assert(data != NULL);



        monitor_json_emitter(mon, data);

        qobject_decref(data);

    }

}
