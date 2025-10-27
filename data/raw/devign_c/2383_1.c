static void test_lifecycle(void)

{

    Coroutine *coroutine;

    bool done = false;



    /* Create, enter, and return from coroutine */

    coroutine = qemu_coroutine_create(set_and_exit);

    qemu_coroutine_enter(coroutine, &done);

    g_assert(done); /* expect done to be true (first time) */



    /* Repeat to check that no state affects this test */

    done = false;

    coroutine = qemu_coroutine_create(set_and_exit);

    qemu_coroutine_enter(coroutine, &done);

    g_assert(done); /* expect done to be true (second time) */

}
