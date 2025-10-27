static GThread *trace_thread_create(GThreadFunc fn)

{

    GThread *thread;

#ifndef _WIN32

    sigset_t set, oldset;



    sigfillset(&set);

    pthread_sigmask(SIG_SETMASK, &set, &oldset);

#endif

    thread = g_thread_create(writeout_thread, NULL, FALSE, NULL);

#ifndef _WIN32

    pthread_sigmask(SIG_SETMASK, &oldset, NULL);

#endif



    return thread;

}
