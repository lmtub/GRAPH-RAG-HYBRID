static QDict *parse_json_filename(const char *filename, Error **errp)

{

    QObject *options_obj;

    QDict *options;

    int ret;



    ret = strstart(filename, "json:", &filename);

    assert(ret);



    options_obj = qobject_from_json(filename);

    if (!options_obj) {

        error_setg(errp, "Could not parse the JSON options");

        return NULL;

    }



    if (qobject_type(options_obj) != QTYPE_QDICT) {

        qobject_decref(options_obj);

        error_setg(errp, "Invalid JSON object given");

        return NULL;

    }



    options = qobject_to_qdict(options_obj);

    qdict_flatten(options);



    return options;

}
