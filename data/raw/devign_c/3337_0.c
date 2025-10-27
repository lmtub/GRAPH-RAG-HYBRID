static int check_arg(const CmdArgs *cmd_args, QDict *args)

{

    QObject *value;

    const char *name;



    name = qstring_get_str(cmd_args->name);



    if (!args) {

        return check_opt(cmd_args, name, args);

    }



    value = qdict_get(args, name);

    if (!value) {

        return check_opt(cmd_args, name, args);

    }



    switch (cmd_args->type) {

        case 'F':

        case 'B':

        case 's':

            if (qobject_type(value) != QTYPE_QSTRING) {

                qerror_report(QERR_INVALID_PARAMETER_TYPE, name, "string");

                return -1;

            }

            break;

        case '/': {

            int i;

            const char *keys[] = { "count", "format", "size", NULL };



            for (i = 0; keys[i]; i++) {

                QObject *obj = qdict_get(args, keys[i]);

                if (!obj) {

                    qerror_report(QERR_MISSING_PARAMETER, name);

                    return -1;

                }

                if (qobject_type(obj) != QTYPE_QINT) {

                    qerror_report(QERR_INVALID_PARAMETER_TYPE, name, "int");

                    return -1;

                }

            }

            break;

        }

        case 'i':

        case 'l':

        case 'M':

            if (qobject_type(value) != QTYPE_QINT) {

                qerror_report(QERR_INVALID_PARAMETER_TYPE, name, "int");

                return -1;

            }

            break;

        case 'f':

        case 'T':

            if (qobject_type(value) != QTYPE_QINT && qobject_type(value) != QTYPE_QFLOAT) {

                qerror_report(QERR_INVALID_PARAMETER_TYPE, name, "number");

                return -1;

            }

            break;

        case 'b':

            if (qobject_type(value) != QTYPE_QBOOL) {

                qerror_report(QERR_INVALID_PARAMETER_TYPE, name, "bool");

                return -1;

            }

            break;

        case '-':

            if (qobject_type(value) != QTYPE_QINT &&

                qobject_type(value) != QTYPE_QBOOL) {

                qerror_report(QERR_INVALID_PARAMETER_TYPE, name, "bool");

                return -1;

            }

            break;

        case 'O':

        default:

            /* impossible */

            abort();

    }



    return 0;

}
