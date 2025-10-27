void error_set_field(Error *err, const char *field, const char *value)

{

    QDict *dict = qdict_get_qdict(err->obj, "data");

    return qdict_put(dict, field, qstring_from_str(value));

}
