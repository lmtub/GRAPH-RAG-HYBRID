static void qdict_array_split_test(void)

{

    QDict *test_dict = qdict_new();

    QDict *dict1, *dict2;

    QList *test_list;



    /*

     * Test the split of

     *

     * {

     *     "1.x": 0,

     *     "3.y": 1,

     *     "0.a": 42,

     *     "o.o": 7,

     *     "0.b": 23

     * }

     *

     * to

     *

     * [

     *     {

     *         "a": 42,

     *         "b": 23

     *     },

     *     {

     *         "x": 0

     *     }

     * ]

     *

     * and

     *

     * {

     *     "3.y": 1,

     *     "o.o": 7

     * }

     *

     * (remaining in the old QDict)

     *

     * This example is given in the comment of qdict_array_split().

     */



    qdict_put(test_dict, "1.x", qint_from_int(0));

    qdict_put(test_dict, "3.y", qint_from_int(1));

    qdict_put(test_dict, "0.a", qint_from_int(42));

    qdict_put(test_dict, "o.o", qint_from_int(7));

    qdict_put(test_dict, "0.b", qint_from_int(23));



    qdict_array_split(test_dict, &test_list);



    dict1 = qobject_to_qdict(qlist_pop(test_list));

    dict2 = qobject_to_qdict(qlist_pop(test_list));



    g_assert(dict1);

    g_assert(dict2);

    g_assert(qlist_empty(test_list));



    QDECREF(test_list);



    g_assert(qdict_get_int(dict1, "a") == 42);

    g_assert(qdict_get_int(dict1, "b") == 23);



    g_assert(qdict_size(dict1) == 2);



    QDECREF(dict1);



    g_assert(qdict_get_int(dict2, "x") == 0);



    g_assert(qdict_size(dict2) == 1);



    QDECREF(dict2);



    g_assert(qdict_get_int(test_dict, "3.y") == 1);

    g_assert(qdict_get_int(test_dict, "o.o") == 7);



    g_assert(qdict_size(test_dict) == 2);



    QDECREF(test_dict);

}
