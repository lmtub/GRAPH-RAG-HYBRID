const char *qdict_get_str(const QDict *qdict, const char *key)

{

    QObject *obj = qdict_get_obj(qdict, key, QTYPE_QSTRING);

    return qstring_get_str(qobject_to_qstring(obj));

}
