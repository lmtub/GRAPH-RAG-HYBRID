static void test_nesting(void)

{

    Coroutine *root;

    NestData nd = {

        .n_enter  = 0,

        .n_return = 0,

        .max      = 128,

    };



    root = qemu_coroutine_create(nest);

    qemu_coroutine_enter(root, &nd);



    /* Must enter and return from max nesting level */

    g_assert_cmpint(nd.n_enter, ==, nd.max);

    g_assert_cmpint(nd.n_return, ==, nd.max);

}
