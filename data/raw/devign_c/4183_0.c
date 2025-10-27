int target_to_host_signal(int sig)

{

    if (sig >= _NSIG)

        return sig;

    return target_to_host_signal_table[sig];

}
