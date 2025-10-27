static unsigned __stdcall win32_start_routine(void *arg)

{

    struct QemuThreadData data = *(struct QemuThreadData *) arg;

    QemuThread *thread = data.thread;



    free(arg);

    TlsSetValue(qemu_thread_tls_index, thread);



    /*

     * Use DuplicateHandle instead of assigning thread->thread in the

     * creating thread to avoid races.  It's simpler this way than with

     * synchronization.

     */

    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),

                    GetCurrentProcess(), &thread->thread,

                    0, FALSE, DUPLICATE_SAME_ACCESS);



    qemu_thread_exit(data.start_routine(data.arg));

    abort();

}
