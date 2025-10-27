int qemu_thread_is_self(QemuThread *thread)

{

    QemuThread *this_thread = TlsGetValue(qemu_thread_tls_index);

    return this_thread->thread == thread->thread;

}
