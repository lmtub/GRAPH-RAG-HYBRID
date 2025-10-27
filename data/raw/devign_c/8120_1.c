int64_t qdict_get_int(const QDict *qdict, const char *key)

{

    QObject *obj = qdict_get_obj(qdict, key, QTYPE_QINT);

    return qint_get_int(qobject_to_qint(obj));

}
