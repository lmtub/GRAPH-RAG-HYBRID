void optimize_flags_init(void)

{

    cpu_env = tcg_global_reg_new_ptr(TCG_AREG0, "env");

    cpu_cc_op = tcg_global_mem_new_i32(TCG_AREG0,

                                       offsetof(CPUX86State, cc_op), "cc_op");

    cpu_cc_src = tcg_global_mem_new(TCG_AREG0, offsetof(CPUX86State, cc_src),

                                    "cc_src");

    cpu_cc_dst = tcg_global_mem_new(TCG_AREG0, offsetof(CPUX86State, cc_dst),

                                    "cc_dst");



#ifdef TARGET_X86_64

    cpu_regs[R_EAX] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EAX]), "rax");

    cpu_regs[R_ECX] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_ECX]), "rcx");

    cpu_regs[R_EDX] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EDX]), "rdx");

    cpu_regs[R_EBX] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EBX]), "rbx");

    cpu_regs[R_ESP] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_ESP]), "rsp");

    cpu_regs[R_EBP] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EBP]), "rbp");

    cpu_regs[R_ESI] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_ESI]), "rsi");

    cpu_regs[R_EDI] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EDI]), "rdi");

    cpu_regs[8] = tcg_global_mem_new_i64(TCG_AREG0,

                                         offsetof(CPUX86State, regs[8]), "r8");

    cpu_regs[9] = tcg_global_mem_new_i64(TCG_AREG0,

                                          offsetof(CPUX86State, regs[9]), "r9");

    cpu_regs[10] = tcg_global_mem_new_i64(TCG_AREG0,

                                          offsetof(CPUX86State, regs[10]), "r10");

    cpu_regs[11] = tcg_global_mem_new_i64(TCG_AREG0,

                                          offsetof(CPUX86State, regs[11]), "r11");

    cpu_regs[12] = tcg_global_mem_new_i64(TCG_AREG0,

                                          offsetof(CPUX86State, regs[12]), "r12");

    cpu_regs[13] = tcg_global_mem_new_i64(TCG_AREG0,

                                          offsetof(CPUX86State, regs[13]), "r13");

    cpu_regs[14] = tcg_global_mem_new_i64(TCG_AREG0,

                                          offsetof(CPUX86State, regs[14]), "r14");

    cpu_regs[15] = tcg_global_mem_new_i64(TCG_AREG0,

                                          offsetof(CPUX86State, regs[15]), "r15");

#else

    cpu_regs[R_EAX] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EAX]), "eax");

    cpu_regs[R_ECX] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_ECX]), "ecx");

    cpu_regs[R_EDX] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EDX]), "edx");

    cpu_regs[R_EBX] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EBX]), "ebx");

    cpu_regs[R_ESP] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_ESP]), "esp");

    cpu_regs[R_EBP] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EBP]), "ebp");

    cpu_regs[R_ESI] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_ESI]), "esi");

    cpu_regs[R_EDI] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUX86State, regs[R_EDI]), "edi");

#endif



    /* register helpers */

#define GEN_HELPER 2

#include "helper.h"

}
