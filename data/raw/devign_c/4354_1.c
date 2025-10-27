void helper_wrpsr(CPUSPARCState *env, target_ulong new_psr)

{

    if ((new_psr & PSR_CWP) >= env->nwindows) {

        cpu_raise_exception_ra(env, TT_ILL_INSN, GETPC());

    } else {



        cpu_put_psr(env, new_psr);


    }

}