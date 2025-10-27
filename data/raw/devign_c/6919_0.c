static void test_dealloc_types(void)

{

    UserDefOne *ud1test, *ud1a, *ud1b;

    UserDefOneList *ud1list;



    ud1test = g_malloc0(sizeof(UserDefOne));

    ud1test->base = g_new0(UserDefZero, 1);

    ud1test->base->integer = 42;

    ud1test->string = g_strdup("hi there 42");



    qapi_free_UserDefOne(ud1test);



    ud1a = g_malloc0(sizeof(UserDefOne));

    ud1a->base = g_new0(UserDefZero, 1);

    ud1a->base->integer = 43;

    ud1a->string = g_strdup("hi there 43");



    ud1b = g_malloc0(sizeof(UserDefOne));

    ud1b->base = g_new0(UserDefZero, 1);

    ud1b->base->integer = 44;

    ud1b->string = g_strdup("hi there 44");



    ud1list = g_malloc0(sizeof(UserDefOneList));

    ud1list->value = ud1a;

    ud1list->next = g_malloc0(sizeof(UserDefOneList));

    ud1list->next->value = ud1b;



    qapi_free_UserDefOneList(ud1list);

}
