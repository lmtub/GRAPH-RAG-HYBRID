static void check_native_list(QObject *qobj,

                              UserDefNativeListUnionKind kind)

{

    QDict *qdict;

    QList *qlist;

    int i;



    g_assert(qobj);

    g_assert(qobject_type(qobj) == QTYPE_QDICT);

    qdict = qobject_to_qdict(qobj);

    g_assert(qdict);

    g_assert(qdict_haskey(qdict, "data"));

    qlist = qlist_copy(qobject_to_qlist(qdict_get(qdict, "data")));



    switch (kind) {

    case USER_DEF_NATIVE_LIST_UNION_KIND_S8:

    case USER_DEF_NATIVE_LIST_UNION_KIND_S16:

    case USER_DEF_NATIVE_LIST_UNION_KIND_S32:

    case USER_DEF_NATIVE_LIST_UNION_KIND_S64:

    case USER_DEF_NATIVE_LIST_UNION_KIND_U8:

    case USER_DEF_NATIVE_LIST_UNION_KIND_U16:

    case USER_DEF_NATIVE_LIST_UNION_KIND_U32:

    case USER_DEF_NATIVE_LIST_UNION_KIND_U64:

        /* all integer elements in JSON arrays get stored into QInts when

         * we convert to QObjects, so we can check them all in the same

         * fashion, so simply fall through here

         */

    case USER_DEF_NATIVE_LIST_UNION_KIND_INTEGER:

        for (i = 0; i < 32; i++) {

            QObject *tmp;

            QInt *qvalue;

            tmp = qlist_peek(qlist);

            g_assert(tmp);

            qvalue = qobject_to_qint(tmp);

            g_assert_cmpint(qint_get_int(qvalue), ==, i);

            qobject_decref(qlist_pop(qlist));

        }

        break;

    case USER_DEF_NATIVE_LIST_UNION_KIND_BOOLEAN:

        for (i = 0; i < 32; i++) {

            QObject *tmp;

            QBool *qvalue;

            tmp = qlist_peek(qlist);

            g_assert(tmp);

            qvalue = qobject_to_qbool(tmp);

            g_assert_cmpint(qbool_get_bool(qvalue), ==, i % 3 == 0);

            qobject_decref(qlist_pop(qlist));

        }

        break;

    case USER_DEF_NATIVE_LIST_UNION_KIND_STRING:

        for (i = 0; i < 32; i++) {

            QObject *tmp;

            QString *qvalue;

            gchar str[8];

            tmp = qlist_peek(qlist);

            g_assert(tmp);

            qvalue = qobject_to_qstring(tmp);

            sprintf(str, "%d", i);

            g_assert_cmpstr(qstring_get_str(qvalue), ==, str);

            qobject_decref(qlist_pop(qlist));

        }

        break;

    case USER_DEF_NATIVE_LIST_UNION_KIND_NUMBER:

        for (i = 0; i < 32; i++) {

            QObject *tmp;

            QFloat *qvalue;

            GString *double_expected = g_string_new("");

            GString *double_actual = g_string_new("");



            tmp = qlist_peek(qlist);

            g_assert(tmp);

            qvalue = qobject_to_qfloat(tmp);

            g_string_printf(double_expected, "%.6f", (double)i / 3);

            g_string_printf(double_actual, "%.6f", qfloat_get_double(qvalue));

            g_assert_cmpstr(double_actual->str, ==, double_expected->str);



            qobject_decref(qlist_pop(qlist));

            g_string_free(double_expected, true);

            g_string_free(double_actual, true);

        }

        break;

    default:

        g_assert_not_reached();

    }

    QDECREF(qlist);

}
