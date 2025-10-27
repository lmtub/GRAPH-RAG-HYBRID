START_TEST(qfloat_destroy_test)

{

    QFloat *qf = qfloat_from_double(0.0);

    QDECREF(qf);

}
