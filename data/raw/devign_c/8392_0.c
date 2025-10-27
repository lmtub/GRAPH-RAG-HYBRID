static void coroutine_enter_cb(void *opaque, int ret)

{

    Coroutine *co = opaque;

    qemu_coroutine_enter(co, NULL);

}
