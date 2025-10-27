static inline int copy_siginfo_to_user(target_siginfo_t *tinfo,

                                       const target_siginfo_t *info)

{

    tswap_siginfo(tinfo, info);

    return 0;

}
