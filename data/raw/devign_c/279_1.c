QDict *qdict_get_qdict(const QDict *qdict, const char *key)

{

    return qobject_to_qdict(qdict_get_obj(qdict, key, QTYPE_QDICT));

}
