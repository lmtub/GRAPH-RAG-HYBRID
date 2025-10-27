static void coroutine_fn verify_self(void *opaque)

{

    g_assert(qemu_coroutine_self() == opaque);

}
