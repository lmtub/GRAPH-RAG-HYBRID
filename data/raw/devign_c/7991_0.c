void helper_lret_protected(int shift, int addend)

{

    helper_ret_protected(shift, 0, addend);

#ifdef CONFIG_KQEMU

    if (kqemu_is_ok(env)) {

        env->exception_index = -1;

        cpu_loop_exit();

    }

#endif

}
