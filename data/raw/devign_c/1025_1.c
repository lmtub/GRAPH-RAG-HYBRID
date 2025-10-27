static void error_callback_bh(void *opaque)

{

    Coroutine *co = opaque;

    qemu_coroutine_enter(co);

}
