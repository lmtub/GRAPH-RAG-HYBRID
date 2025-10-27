static void coroutine_fn verify_entered_step_2(void *opaque)

{

    Coroutine *caller = (Coroutine *)opaque;



    g_assert(qemu_coroutine_entered(caller));

    g_assert(qemu_coroutine_entered(qemu_coroutine_self()));

    qemu_coroutine_yield();



    /* Once more to check it still works after yielding */

    g_assert(qemu_coroutine_entered(caller));

    g_assert(qemu_coroutine_entered(qemu_coroutine_self()));

    qemu_coroutine_yield();

}
