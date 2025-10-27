int qemu_get_thread_id(void)

{

#if defined (__linux__)

    return syscall(SYS_gettid);

#else

    return getpid();

#endif

}
