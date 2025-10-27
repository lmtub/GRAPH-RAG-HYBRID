static void simple_number(void)

{

    int i;

    struct {

        const char *encoded;

        int64_t decoded;

        int skip;

    } test_cases[] = {

        { "0", 0 },

        { "1234", 1234 },

        { "1", 1 },

        { "-32", -32 },

        { "-0", 0, .skip = 1 },

        { },

    };



    for (i = 0; test_cases[i].encoded; i++) {

        QInt *qint;



        qint = qobject_to_qint(qobject_from_json(test_cases[i].encoded, NULL));

        g_assert(qint);

        g_assert(qint_get_int(qint) == test_cases[i].decoded);

        if (test_cases[i].skip == 0) {

            QString *str;



            str = qobject_to_json(QOBJECT(qint));

            g_assert(strcmp(qstring_get_str(str), test_cases[i].encoded) == 0);

            QDECREF(str);

        }



        QDECREF(qint);

    }

}
