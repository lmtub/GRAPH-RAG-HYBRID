static void qemu_thread_set_name(QemuThread *thread, const char *name)

{

#ifdef CONFIG_PTHREAD_SETNAME_NP

    pthread_setname_np(thread->thread, name);

#endif

}
