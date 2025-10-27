static JSONTokenType token_get_type(QObject *obj)

{

    return qdict_get_int(qobject_to_qdict(obj), "type");

}
