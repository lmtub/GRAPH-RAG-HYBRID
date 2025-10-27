static int do_qmp_capabilities(Monitor *mon, const QDict *params,

                               QObject **ret_data)

{

    /* Will setup QMP capabilities in the future */

    if (monitor_ctrl_mode(mon)) {

        mon->qmp.command_mode = 1;

    }



    return 0;

}
