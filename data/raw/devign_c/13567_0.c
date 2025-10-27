static uint64_t pmsav7_read(CPUARMState *env, const ARMCPRegInfo *ri)

{

    uint32_t *u32p = *(uint32_t **)raw_ptr(env, ri);



    if (!u32p) {

        return 0;

    }



    u32p += env->pmsav7.rnr;

    return *u32p;

}
