static QDict *qmp_check_input_obj(QObject *input_obj)

{

    const QDictEntry *ent;

    int has_exec_key = 0;

    QDict *input_dict;



    if (qobject_type(input_obj) != QTYPE_QDICT) {

        qerror_report(QERR_QMP_BAD_INPUT_OBJECT, "object");

        return NULL;

    }



    input_dict = qobject_to_qdict(input_obj);



    for (ent = qdict_first(input_dict); ent; ent = qdict_next(input_dict, ent)){

        const char *arg_name = qdict_entry_key(ent);

        const QObject *arg_obj = qdict_entry_value(ent);



        if (!strcmp(arg_name, "execute")) {

            if (qobject_type(arg_obj) != QTYPE_QSTRING) {

                qerror_report(QERR_QMP_BAD_INPUT_OBJECT_MEMBER, "execute",

                              "string");

                return NULL;

            }

            has_exec_key = 1;

        } else if (!strcmp(arg_name, "arguments")) {

            if (qobject_type(arg_obj) != QTYPE_QDICT) {

                qerror_report(QERR_QMP_BAD_INPUT_OBJECT_MEMBER, "arguments",

                              "object");

                return NULL;

            }

        } else if (!strcmp(arg_name, "id")) {

            /* FIXME: check duplicated IDs for async commands */

        } else {

            qerror_report(QERR_QMP_EXTRA_MEMBER, arg_name);

            return NULL;

        }

    }



    if (!has_exec_key) {

        qerror_report(QERR_QMP_BAD_INPUT_OBJECT, "execute");

        return NULL;

    }



    return input_dict;

}
