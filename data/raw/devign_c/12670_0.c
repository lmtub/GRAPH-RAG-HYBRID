static void monitor_call_handler(Monitor *mon, const mon_cmd_t *cmd,

                                 const QDict *params)

{

    QObject *data = NULL;



    cmd->mhandler.cmd_new(mon, params, &data);

    if (data)

        cmd->user_print(mon, data);



    qobject_decref(data);

}
