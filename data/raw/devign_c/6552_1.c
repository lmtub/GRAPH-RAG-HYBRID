static void handle_qmp_command(JSONMessageParser *parser, GQueue *tokens)

{

    QObject *req, *rsp = NULL, *id = NULL;

    QDict *qdict = NULL;

    const char *cmd_name;

    Monitor *mon = cur_mon;

    Error *err = NULL;



    req = json_parser_parse_err(tokens, NULL, &err);

    if (!req && !err) {

        /* json_parser_parse_err() sucks: can fail without setting @err */

        error_setg(&err, QERR_JSON_PARSING);

    }

    if (err) {

        goto err_out;

    }



    qdict = qmp_check_input_obj(req, &err);

    if (!qdict) {

        goto err_out;

    }



    id = qdict_get(qdict, "id");

    qobject_incref(id);

    qdict_del(qdict, "id");



    cmd_name = qdict_get_str(qdict, "execute");

    trace_handle_qmp_command(mon, cmd_name);



    rsp = qmp_dispatch(cur_mon->qmp.commands, req);



    if (mon->qmp.commands == &qmp_cap_negotiation_commands) {

        qdict = qdict_get_qdict(qobject_to_qdict(rsp), "error");

        if (qdict

            && !g_strcmp0(qdict_get_try_str(qdict, "class"),

                    QapiErrorClass_lookup[ERROR_CLASS_COMMAND_NOT_FOUND])) {

            /* Provide a more useful error message */

            qdict_del(qdict, "desc");

            qdict_put(qdict, "desc",

                      qstring_from_str("Expecting capabilities negotiation"

                                       " with 'qmp_capabilities'"));

        }

    }



err_out:

    if (err) {

        qdict = qdict_new();

        qdict_put_obj(qdict, "error", qmp_build_error_object(err));

        error_free(err);

        rsp = QOBJECT(qdict);

    }



    if (rsp) {

        if (id) {

            qdict_put_obj(qobject_to_qdict(rsp), "id", id);

            id = NULL;

        }



        monitor_json_emitter(mon, rsp);

    }



    qobject_decref(id);

    qobject_decref(rsp);

    qobject_decref(req);

}
