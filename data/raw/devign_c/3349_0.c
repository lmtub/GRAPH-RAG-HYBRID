void HELPER(wsr_ibreaka)(uint32_t i, uint32_t v)

{

    if (env->sregs[IBREAKENABLE] & (1 << i) && env->sregs[IBREAKA + i] != v) {

        tb_invalidate_phys_page_range(

                env->sregs[IBREAKA + i], env->sregs[IBREAKA + i] + 1, 0);

        tb_invalidate_phys_page_range(v, v + 1, 0);

    }

    env->sregs[IBREAKA + i] = v;

}
