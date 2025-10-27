QBool *qbool_from_bool(bool value)

{

    QBool *qb;



    qb = g_malloc(sizeof(*qb));

    qb->value = value;

    QOBJECT_INIT(qb, &qbool_type);



    return qb;

}
