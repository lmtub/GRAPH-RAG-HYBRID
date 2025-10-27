void do_mouse_set(Monitor *mon, const QDict *qdict)

{

    QemuInputHandlerState *s;

    int index = qdict_get_int(qdict, "index");

    int found = 0;



    QTAILQ_FOREACH(s, &handlers, node) {

        if (s->id == index) {

            found = 1;

            qemu_input_handler_activate(s);

            break;

        }

    }



    if (!found) {

        monitor_printf(mon, "Mouse at given index not found\n");

    }



    qemu_input_check_mode_change();

}
