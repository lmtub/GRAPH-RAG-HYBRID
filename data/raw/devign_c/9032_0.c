START_TEST(simple_list)

{

    int i;

    struct {

        const char *encoded;

        LiteralQObject decoded;

    } test_cases[] = {

        {

            .encoded = "[43,42]",

            .decoded = QLIT_QLIST(((LiteralQObject[]){

                        QLIT_QINT(43),

                        QLIT_QINT(42),

                        { }

                    })),

        },

        {

            .encoded = "[43]",

            .decoded = QLIT_QLIST(((LiteralQObject[]){

                        QLIT_QINT(43),

                        { }

                    })),

        },

        {

            .encoded = "[]",

            .decoded = QLIT_QLIST(((LiteralQObject[]){

                        { }

                    })),

        },

        {

            .encoded = "[{}]",

            .decoded = QLIT_QLIST(((LiteralQObject[]){

                        QLIT_QDICT(((LiteralQDictEntry[]){

                                    {},

                                        })),

                        {},

                            })),

        },

        { }

    };



    for (i = 0; test_cases[i].encoded; i++) {

        QObject *obj;

        QString *str;



        obj = qobject_from_json(test_cases[i].encoded);

        fail_unless(obj != NULL);

        fail_unless(qobject_type(obj) == QTYPE_QLIST);



        fail_unless(compare_litqobj_to_qobj(&test_cases[i].decoded, obj) == 1);



        str = qobject_to_json(obj);

        qobject_decref(obj);



        obj = qobject_from_json(qstring_get_str(str));

        fail_unless(obj != NULL);

        fail_unless(qobject_type(obj) == QTYPE_QLIST);



        fail_unless(compare_litqobj_to_qobj(&test_cases[i].decoded, obj) == 1);

        qobject_decref(obj);

        QDECREF(str);

    }

}
