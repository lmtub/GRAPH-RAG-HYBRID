static QDict *qmp_dispatch_check_obj(const QObject *request, Error **errp)

{

    const QDictEntry *ent;

    const char *arg_name;

    const QObject *arg_obj;

    bool has_exec_key = false;

    QDict *dict = NULL;



    dict = qobject_to_qdict(request);

    if (!dict) {

        error_setg(errp, QERR_QMP_BAD_INPUT_OBJECT,

                   "request is not a dictionary");

        return NULL;

    }



    for (ent = qdict_first(dict); ent;

         ent = qdict_next(dict, ent)) {

        arg_name = qdict_entry_key(ent);

        arg_obj = qdict_entry_value(ent);



        if (!strcmp(arg_name, "execute")) {

            if (qobject_type(arg_obj) != QTYPE_QSTRING) {

                error_setg(errp, QERR_QMP_BAD_INPUT_OBJECT_MEMBER, "execute",

                           "string");

                return NULL;

            }

            has_exec_key = true;

        } else if (!strcmp(arg_name, "arguments")) {

            if (qobject_type(arg_obj) != QTYPE_QDICT) {

                error_setg(errp, QERR_QMP_BAD_INPUT_OBJECT_MEMBER,

                           "arguments", "object");

                return NULL;

            }

        } else {

            error_setg(errp, QERR_QMP_EXTRA_MEMBER, arg_name);

            return NULL;

        }

    }



    if (!has_exec_key) {

        error_setg(errp, QERR_QMP_BAD_INPUT_OBJECT, "execute");

        return NULL;

    }



    return dict;

}
