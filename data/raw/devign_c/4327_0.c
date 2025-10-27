void kqemu_cpu_interrupt(CPUState *env)

{

#if defined(_WIN32)

    /* cancelling the I/O request causes KQEMU to finish executing the

       current block and successfully returning. */

    CancelIo(kqemu_fd);

#endif

}
