static void handle_user_command(Monitor *mon, const char *cmdline)

{

    QDict *qdict;

    const mon_cmd_t *cmd;



    qdict = qdict_new();



    cmd = monitor_parse_command(mon, cmdline, 0, mon->cmd_table, qdict);

    if (!cmd)

        goto out;



    if (handler_is_async(cmd)) {

        user_async_cmd_handler(mon, cmd, qdict);

    } else if (handler_is_qobject(cmd)) {

        QObject *data = NULL;



        /* XXX: ignores the error code */

        cmd->mhandler.cmd_new(mon, qdict, &data);

        assert(!monitor_has_error(mon));

        if (data) {

            cmd->user_print(mon, data);

            qobject_decref(data);

        }

    } else {

        cmd->mhandler.cmd(mon, qdict);

    }



out:

    QDECREF(qdict);

}
