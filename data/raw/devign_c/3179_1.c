static void simple_whitespace(void)

{

    int i;

    struct {

        const char *encoded;

        LiteralQObject decoded;

    } test_cases[] = {

        {

            .encoded = " [ 43 , 42 ]",

            .decoded = QLIT_QLIST(((LiteralQObject[]){

                        QLIT_QINT(43),

                        QLIT_QINT(42),

                        { }

                    })),

        },

        {

            .encoded = " [ 43 , { 'h' : 'b' }, [ ], 42 ]",

            .decoded = QLIT_QLIST(((LiteralQObject[]){

                        QLIT_QINT(43),

                        QLIT_QDICT(((LiteralQDictEntry[]){

                                    { "h", QLIT_QSTR("b") },

                                    { }})),

                        QLIT_QLIST(((LiteralQObject[]){

                                    { }})),

                        QLIT_QINT(42),

                        { }

                    })),

        },

        {

            .encoded = " [ 43 , { 'h' : 'b' , 'a' : 32 }, [ ], 42 ]",

            .decoded = QLIT_QLIST(((LiteralQObject[]){

                        QLIT_QINT(43),

                        QLIT_QDICT(((LiteralQDictEntry[]){

                                    { "h", QLIT_QSTR("b") },

                                    { "a", QLIT_QINT(32) },

                                    { }})),

                        QLIT_QLIST(((LiteralQObject[]){

                                    { }})),

                        QLIT_QINT(42),

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
