static CoroutineThreadState *coroutine_get_thread_state(void)

{

    CoroutineThreadState *s = pthread_getspecific(thread_state_key);



    if (!s) {

        s = g_malloc0(sizeof(*s));

        s->current = &s->leader.base;

        QLIST_INIT(&s->pool);

        pthread_setspecific(thread_state_key, s);

    }

    return s;

}
