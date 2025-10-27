QInt *qint_from_int(int64_t value)

{

    QInt *qi;



    qi = g_malloc(sizeof(*qi));

    qi->value = value;

    QOBJECT_INIT(qi, &qint_type);



    return qi;

}
