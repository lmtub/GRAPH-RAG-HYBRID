QFloat *qfloat_from_double(double value)

{

    QFloat *qf;



    qf = g_malloc(sizeof(*qf));

    qf->value = value;

    QOBJECT_INIT(qf, &qfloat_type);



    return qf;

}
