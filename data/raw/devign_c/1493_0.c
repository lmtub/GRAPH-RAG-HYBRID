static void test_visitor_out_number(TestOutputVisitorData *data,

                                    const void *unused)

{

    double value = 3.14;

    QObject *obj;



    visit_type_number(data->ov, NULL, &value, &error_abort);



    obj = visitor_get(data);

    g_assert(qobject_type(obj) == QTYPE_QFLOAT);

    g_assert(qfloat_get_double(qobject_to_qfloat(obj)) == value);

}
