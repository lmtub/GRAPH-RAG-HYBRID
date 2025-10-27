CPUState *cpu_mb_init (const char *cpu_model)

{

    CPUState *env;

    static int tcg_initialized = 0;

    int i;



    env = qemu_mallocz(sizeof(CPUState));



    cpu_exec_init(env);

    cpu_reset(env);



    env->pvr.regs[0] = PVR0_PVR_FULL_MASK \

                       | PVR0_USE_BARREL_MASK \

                       | PVR0_USE_DIV_MASK \

                       | PVR0_USE_HW_MUL_MASK \

                       | PVR0_USE_EXC_MASK \

                       | PVR0_USE_ICACHE_MASK \

                       | PVR0_USE_DCACHE_MASK \

                       | PVR0_USE_MMU \

                       | (0xb << 8);

    env->pvr.regs[2] = PVR2_D_OPB_MASK \

                        | PVR2_D_LMB_MASK \

                        | PVR2_I_OPB_MASK \

                        | PVR2_I_LMB_MASK \

                        | PVR2_USE_MSR_INSTR \

                        | PVR2_USE_PCMP_INSTR \

                        | PVR2_USE_BARREL_MASK \

                        | PVR2_USE_DIV_MASK \

                        | PVR2_USE_HW_MUL_MASK \

                        | PVR2_USE_MUL64_MASK \

                        | 0;

    env->pvr.regs[10] = 0x0c000000; /* Default to spartan 3a dsp family.  */

    env->pvr.regs[11] = PVR11_USE_MMU | (16 << 17);

#if !defined(CONFIG_USER_ONLY)

    env->mmu.c_mmu = 3;

    env->mmu.c_mmu_tlb_access = 3;

    env->mmu.c_mmu_zones = 16;

#endif



    if (tcg_initialized)

        return env;



    tcg_initialized = 1;



    cpu_env = tcg_global_reg_new_ptr(TCG_AREG0, "env");



    env_debug = tcg_global_mem_new(TCG_AREG0, 

                    offsetof(CPUState, debug),

                    "debug0");

    env_iflags = tcg_global_mem_new(TCG_AREG0, 

                    offsetof(CPUState, iflags),

                    "iflags");

    env_imm = tcg_global_mem_new(TCG_AREG0, 

                    offsetof(CPUState, imm),

                    "imm");

    env_btarget = tcg_global_mem_new(TCG_AREG0,

                     offsetof(CPUState, btarget),

                     "btarget");

    env_btaken = tcg_global_mem_new(TCG_AREG0,

                     offsetof(CPUState, btaken),

                     "btaken");

    for (i = 0; i < ARRAY_SIZE(cpu_R); i++) {

        cpu_R[i] = tcg_global_mem_new(TCG_AREG0,

                          offsetof(CPUState, regs[i]),

                          regnames[i]);

    }

    for (i = 0; i < ARRAY_SIZE(cpu_SR); i++) {

        cpu_SR[i] = tcg_global_mem_new(TCG_AREG0,

                          offsetof(CPUState, sregs[i]),

                          special_regnames[i]);

    }

#define GEN_HELPER 2

#include "helper.h"



    return env;

}
