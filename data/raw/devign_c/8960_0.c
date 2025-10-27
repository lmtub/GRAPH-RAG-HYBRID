static void qemu_coroutine_thread_cleanup(void *opaque)

{

    CoroutineThreadState *s = opaque;

    Coroutine *co;

    Coroutine *tmp;



    QLIST_FOREACH_SAFE(co, &s->pool, pool_next, tmp) {

        g_free(DO_UPCAST(CoroutineUContext, base, co)->stack);

        g_free(co);

    }

    g_free(s);

}
