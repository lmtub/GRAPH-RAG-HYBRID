void gen_pc_load(CPUState *env, TranslationBlock *tb,

                unsigned long searched_pc, int pc_pos, void *puc)

{

    env->regs[15] = gen_opc_pc[pc_pos];


}