static void test_in_coroutine(void)

{

    Coroutine *coroutine;



    g_assert(!qemu_in_coroutine());



    coroutine = qemu_coroutine_create(verify_in_coroutine);

    qemu_coroutine_enter(coroutine, NULL);

}
