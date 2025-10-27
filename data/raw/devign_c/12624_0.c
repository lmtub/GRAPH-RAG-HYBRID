static void do_info_commands(Monitor *mon, QObject **ret_data)

{

    QList *cmd_list;

    const mon_cmd_t *cmd;



    cmd_list = qlist_new();



    for (cmd = qmp_cmds; cmd->name != NULL; cmd++) {

        if (monitor_handler_ported(cmd) && !monitor_cmd_user_only(cmd) &&

            !compare_cmd(cmd->name, "info")) {

            qlist_append_obj(cmd_list, get_cmd_dict(cmd->name));

        }

    }



    for (cmd = qmp_query_cmds; cmd->name != NULL; cmd++) {

        if (monitor_handler_ported(cmd) && !monitor_cmd_user_only(cmd)) {

            char buf[128];

            snprintf(buf, sizeof(buf), "query-%s", cmd->name);

            qlist_append_obj(cmd_list, get_cmd_dict(buf));

        }

    }



    *ret_data = QOBJECT(cmd_list);

}
