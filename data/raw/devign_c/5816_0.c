static const char *token_get_value(QObject *obj)

{

    return qdict_get_str(qobject_to_qdict(obj), "token");

}
