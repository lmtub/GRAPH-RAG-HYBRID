void ppc_translate_init(void)

{

    int i;

    char* p;

    size_t cpu_reg_names_size;

    static int done_init = 0;



    if (done_init)

        return;



    cpu_env = tcg_global_reg_new_ptr(TCG_AREG0, "env");



    p = cpu_reg_names;

    cpu_reg_names_size = sizeof(cpu_reg_names);



    for (i = 0; i < 8; i++) {

        snprintf(p, cpu_reg_names_size, "crf%d", i);

        cpu_crf[i] = tcg_global_mem_new_i32(TCG_AREG0,

                                            offsetof(CPUState, crf[i]), p);

        p += 5;

        cpu_reg_names_size -= 5;

    }



    for (i = 0; i < 32; i++) {

        snprintf(p, cpu_reg_names_size, "r%d", i);

        cpu_gpr[i] = tcg_global_mem_new(TCG_AREG0,

                                        offsetof(CPUState, gpr[i]), p);

        p += (i < 10) ? 3 : 4;

        cpu_reg_names_size -= (i < 10) ? 3 : 4;

#if !defined(TARGET_PPC64)

        snprintf(p, cpu_reg_names_size, "r%dH", i);

        cpu_gprh[i] = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUState, gprh[i]), p);

        p += (i < 10) ? 4 : 5;

        cpu_reg_names_size -= (i < 10) ? 4 : 5;

#endif



        snprintf(p, cpu_reg_names_size, "fp%d", i);

        cpu_fpr[i] = tcg_global_mem_new_i64(TCG_AREG0,

                                            offsetof(CPUState, fpr[i]), p);

        p += (i < 10) ? 4 : 5;

        cpu_reg_names_size -= (i < 10) ? 4 : 5;



        snprintf(p, cpu_reg_names_size, "avr%dH", i);

#ifdef HOST_WORDS_BIGENDIAN

        cpu_avrh[i] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUState, avr[i].u64[0]), p);

#else

        cpu_avrh[i] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUState, avr[i].u64[1]), p);

#endif

        p += (i < 10) ? 6 : 7;

        cpu_reg_names_size -= (i < 10) ? 6 : 7;



        snprintf(p, cpu_reg_names_size, "avr%dL", i);

#ifdef HOST_WORDS_BIGENDIAN

        cpu_avrl[i] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUState, avr[i].u64[1]), p);

#else

        cpu_avrl[i] = tcg_global_mem_new_i64(TCG_AREG0,

                                             offsetof(CPUState, avr[i].u64[0]), p);

#endif

        p += (i < 10) ? 6 : 7;

        cpu_reg_names_size -= (i < 10) ? 6 : 7;

    }



    cpu_nip = tcg_global_mem_new(TCG_AREG0,

                                 offsetof(CPUState, nip), "nip");



    cpu_msr = tcg_global_mem_new(TCG_AREG0,

                                 offsetof(CPUState, msr), "msr");



    cpu_ctr = tcg_global_mem_new(TCG_AREG0,

                                 offsetof(CPUState, ctr), "ctr");



    cpu_lr = tcg_global_mem_new(TCG_AREG0,

                                offsetof(CPUState, lr), "lr");



    cpu_xer = tcg_global_mem_new(TCG_AREG0,

                                 offsetof(CPUState, xer), "xer");



    cpu_reserve = tcg_global_mem_new(TCG_AREG0,

                                     offsetof(CPUState, reserve), "reserve");



    cpu_fpscr = tcg_global_mem_new_i32(TCG_AREG0,

                                       offsetof(CPUState, fpscr), "fpscr");



    cpu_access_type = tcg_global_mem_new_i32(TCG_AREG0,

                                             offsetof(CPUState, access_type), "access_type");



    /* register helpers */

#define GEN_HELPER 2

#include "helper.h"



    done_init = 1;

}
