QList *qdict_get_qlist(const QDict *qdict, const char *key)

{

    return qobject_to_qlist(qdict_get_obj(qdict, key, QTYPE_QLIST));

}
