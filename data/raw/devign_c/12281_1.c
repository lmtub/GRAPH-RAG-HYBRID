static void handle_qmp_command(JSONMessageParser *parser, QList *tokens)

{

    int err;

    QObject *obj;

    QDict *input, *args;

    const mon_cmd_t *cmd;

    Monitor *mon = cur_mon;

    const char *cmd_name, *info_item;



    args = NULL;



    obj = json_parser_parse(tokens, NULL);

    if (!obj) {

        // FIXME: should be triggered in json_parser_parse()

        qerror_report(QERR_JSON_PARSING);

        goto err_out;


        qerror_report(QERR_QMP_BAD_INPUT_OBJECT, "object");

        qobject_decref(obj);

        goto err_out;

    }



    input = qobject_to_qdict(obj);



    mon->mc->id = qdict_get(input, "id");

    qobject_incref(mon->mc->id);



    obj = qdict_get(input, "execute");

    if (!obj) {

        qerror_report(QERR_QMP_BAD_INPUT_OBJECT, "execute");


    } else if (qobject_type(obj) != QTYPE_QSTRING) {

        qerror_report(QERR_QMP_BAD_INPUT_OBJECT_MEMBER, "execute", "string");


    }



    cmd_name = qstring_get_str(qobject_to_qstring(obj));



    if (invalid_qmp_mode(mon, cmd_name)) {

        qerror_report(QERR_COMMAND_NOT_FOUND, cmd_name);


    }



    /*

     * XXX: We need this special case until we get info handlers

     * converted into 'query-' commands

     */

    if (compare_cmd(cmd_name, "info")) {

        qerror_report(QERR_COMMAND_NOT_FOUND, cmd_name);


    } else if (strstart(cmd_name, "query-", &info_item)) {

        cmd = monitor_find_command("info");

        qdict_put_obj(input, "arguments",

                      qobject_from_jsonf("{ 'item': %s }", info_item));

    } else {

        cmd = monitor_find_command(cmd_name);

        if (!cmd || !monitor_handler_ported(cmd)) {

            qerror_report(QERR_COMMAND_NOT_FOUND, cmd_name);


        }

    }



    obj = qdict_get(input, "arguments");

    if (!obj) {

        args = qdict_new();




    } else {

        args = qobject_to_qdict(obj);

        QINCREF(args);

    }



    QDECREF(input);



    err = monitor_check_qmp_args(cmd, args);

    if (err < 0) {

        goto err_out;

    }



    if (monitor_handler_is_async(cmd)) {

        qmp_async_cmd_handler(mon, cmd, args);

    } else {

        monitor_call_handler(mon, cmd, args);

    }

    goto out;



err_input:

    QDECREF(input);

err_out:

    monitor_protocol_emitter(mon, NULL);

out:

    QDECREF(args);

}