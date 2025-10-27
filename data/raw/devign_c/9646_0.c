void qemu_thread_self(QemuThread *thread)

{

    thread->thread = pthread_self();

}
