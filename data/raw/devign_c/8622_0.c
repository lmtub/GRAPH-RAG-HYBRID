static void handle_hmp_command(Monitor *mon, const char *cmdline)

{

    QDict *qdict;

    const mon_cmd_t *cmd;



    qdict = qdict_new();



    cmd = monitor_parse_command(mon, cmdline, 0, mon->cmd_table, qdict);

    if (cmd) {

        cmd->mhandler.cmd(mon, qdict);

    }



    QDECREF(qdict);

}
