START_TEST(simple_number)

{

    int i;

    struct {

        const char *encoded;

        int64_t decoded;

    } test_cases[] = {

        { "0", 0 },

        { "1234", 1234 },

        { "1", 1 },

        { "-32", -32 },

        { "-0", 0 },

        { },

    };



    for (i = 0; test_cases[i].encoded; i++) {

        QObject *obj;

        QInt *qint;



        obj = qobject_from_json(test_cases[i].encoded);

        fail_unless(obj != NULL);

        fail_unless(qobject_type(obj) == QTYPE_QINT);



        qint = qobject_to_qint(obj);

        fail_unless(qint_get_int(qint) == test_cases[i].decoded);



        QDECREF(qint);

    }

}
