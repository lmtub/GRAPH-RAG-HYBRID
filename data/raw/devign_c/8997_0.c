static int do_info(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const mon_cmd_t *cmd;

    const char *item = qdict_get_try_str(qdict, "item");



    if (!item) {

        goto help;

    }



    for (cmd = info_cmds; cmd->name != NULL; cmd++) {

        if (compare_cmd(item, cmd->name))

            break;

    }



    if (cmd->name == NULL) {

        goto help;

    }



    if (monitor_handler_is_async(cmd)) {

        user_async_info_handler(mon, cmd);

        /*

         * Indicate that this command is asynchronous and will not return any

         * data (not even empty).  Instead, the data will be returned via a

         * completion callback.

         */

        *ret_data = qobject_from_jsonf("{ '__mon_async': 'return' }");

    } else if (monitor_handler_ported(cmd)) {

        QObject *info_data = NULL;



        cmd->mhandler.info_new(mon, &info_data);

        if (info_data) {

            cmd->user_print(mon, info_data);

            qobject_decref(info_data);

        }

    } else {

        cmd->mhandler.info(mon);

    }



    return 0;



help:

    help_cmd(mon, "info");

    return 0;

}
