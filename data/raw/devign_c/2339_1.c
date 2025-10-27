void host_to_target_siginfo(target_siginfo_t *tinfo, const siginfo_t *info)

{

    host_to_target_siginfo_noswap(tinfo, info);

    tswap_siginfo(tinfo, tinfo);

}
