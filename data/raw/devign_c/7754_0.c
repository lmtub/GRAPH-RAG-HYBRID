QString *qstring_from_substr(const char *str, int start, int end)

{

    QString *qstring;



    qstring = g_malloc(sizeof(*qstring));



    qstring->length = end - start + 1;

    qstring->capacity = qstring->length;



    qstring->string = g_malloc(qstring->capacity + 1);

    memcpy(qstring->string, str + start, qstring->length);

    qstring->string[qstring->length] = 0;



    QOBJECT_INIT(qstring, &qstring_type);



    return qstring;

}
