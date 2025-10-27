int cpu_signal_handler(int host_signum, void *pinfo, void *puc)

{

    siginfo_t *info = pinfo;

    ucontext_t *uc = puc;

    unsigned long ip;

    int is_write = 0;



    ip = uc->uc_mcontext.sc_ip;

    switch (host_signum) {

    case SIGILL:

    case SIGFPE:

    case SIGSEGV:

    case SIGBUS:

    case SIGTRAP:

        if (info->si_code && (info->si_segvflags & __ISR_VALID)) {

            /* ISR.W (write-access) is bit 33:  */

            is_write = (info->si_isr >> 33) & 1;

        }

        break;



    default:

        break;

    }

    return handle_cpu_signal(ip, (unsigned long)info->si_addr,

                             is_write,

                             (sigset_t *)&uc->uc_sigmask);

}
