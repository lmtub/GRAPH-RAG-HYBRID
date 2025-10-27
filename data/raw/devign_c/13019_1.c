static void simple_dict(void)

{

    int i;

    struct {

        const char *encoded;

        LiteralQObject decoded;

    } test_cases[] = {

        {

            .encoded = "{\"foo\": 42, \"bar\": \"hello world\"}",

            .decoded = QLIT_QDICT(((LiteralQDictEntry[]){

                        { "foo", QLIT_QINT(42) },

                        { "bar", QLIT_QSTR("hello world") },

                        { }

                    })),

        }, {

            .encoded = "{}",

            .decoded = QLIT_QDICT(((LiteralQDictEntry[]){

                        { }

                    })),

        }, {

            .encoded = "{\"foo\": 43}",

            .decoded = QLIT_QDICT(((LiteralQDictEntry[]){

                        { "foo", QLIT_QINT(43) },

                        { }

                    })),

        },

        { }

    };



    for (i = 0; test_cases[i].encoded; i++) {

        QObject *obj;

        QString *str;



        obj = qobject_from_json(test_cases[i].encoded, NULL);

        g_assert(compare_litqobj_to_qobj(&test_cases[i].decoded, obj) == 1);



        str = qobject_to_json(obj);

        qobject_decref(obj);



        obj = qobject_from_json(qstring_get_str(str), NULL);

        g_assert(compare_litqobj_to_qobj(&test_cases[i].decoded, obj) == 1);

        qobject_decref(obj);

        QDECREF(str);

    }

}
