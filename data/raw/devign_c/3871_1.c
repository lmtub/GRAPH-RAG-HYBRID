bool qdict_get_bool(const QDict *qdict, const char *key)

{

    QObject *obj = qdict_get_obj(qdict, key, QTYPE_QBOOL);

    return qbool_get_bool(qobject_to_qbool(obj));

}
