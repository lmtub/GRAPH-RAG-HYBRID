void monitor_init(CharDriverState *chr, int flags)

{

    static int is_first_init = 1;

    Monitor *mon;



    if (is_first_init) {

        monitor_qapi_event_init();

        sortcmdlist();

        is_first_init = 0;

    }



    mon = g_malloc(sizeof(*mon));

    monitor_data_init(mon);



    mon->chr = chr;

    mon->flags = flags;

    if (flags & MONITOR_USE_READLINE) {

        mon->rs = readline_init(monitor_readline_printf,

                                monitor_readline_flush,

                                mon,

                                monitor_find_completion);

        monitor_read_command(mon, 0);

    }



    if (monitor_is_qmp(mon)) {

        qemu_chr_add_handlers(chr, monitor_can_read, monitor_qmp_read,

                              monitor_qmp_event, mon);

        qemu_chr_fe_set_echo(chr, true);

        json_message_parser_init(&mon->qmp.parser, handle_qmp_command);

    } else {

        qemu_chr_add_handlers(chr, monitor_can_read, monitor_read,

                              monitor_event, mon);

    }



    qemu_mutex_lock(&monitor_lock);

    QLIST_INSERT_HEAD(&mon_list, mon, entry);

    qemu_mutex_unlock(&monitor_lock);



    if (!default_mon || (flags & MONITOR_IS_DEFAULT))

        default_mon = mon;

}
