static void do_order_test(void)

{

    Coroutine *co;



    co = qemu_coroutine_create(co_order_test);

    record_push(1, 1);

    qemu_coroutine_enter(co, NULL);

    record_push(1, 2);

    g_assert(!qemu_in_coroutine());

    qemu_coroutine_enter(co, NULL);

    record_push(1, 3);

    g_assert(!qemu_in_coroutine());

}
