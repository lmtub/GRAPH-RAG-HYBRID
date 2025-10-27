static int qmp_async_cmd_handler(Monitor *mon, const mon_cmd_t *cmd,

                                 const QDict *params)

{

    return cmd->mhandler.cmd_async(mon, params, qmp_monitor_complete, mon);

}
