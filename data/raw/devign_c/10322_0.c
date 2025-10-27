int qemu_thread_equal(QemuThread *thread1, QemuThread *thread2)

{

   return pthread_equal(thread1->thread, thread2->thread);

}
