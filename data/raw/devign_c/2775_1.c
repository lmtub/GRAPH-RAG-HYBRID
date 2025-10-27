void xtensa_translate_init(void)

{

    static const char * const regnames[] = {

        "ar0", "ar1", "ar2", "ar3",

        "ar4", "ar5", "ar6", "ar7",

        "ar8", "ar9", "ar10", "ar11",

        "ar12", "ar13", "ar14", "ar15",

    };

    static const char * const fregnames[] = {

        "f0", "f1", "f2", "f3",

        "f4", "f5", "f6", "f7",

        "f8", "f9", "f10", "f11",

        "f12", "f13", "f14", "f15",

    };

    int i;



    cpu_env = tcg_global_reg_new_ptr(TCG_AREG0, "env");

    cpu_pc = tcg_global_mem_new_i32(TCG_AREG0,

            offsetof(CPUXtensaState, pc), "pc");



    for (i = 0; i < 16; i++) {

        cpu_R[i] = tcg_global_mem_new_i32(TCG_AREG0,

                offsetof(CPUXtensaState, regs[i]),

                regnames[i]);

    }



    for (i = 0; i < 16; i++) {

        cpu_FR[i] = tcg_global_mem_new_i32(TCG_AREG0,

                offsetof(CPUXtensaState, fregs[i]),

                fregnames[i]);

    }



    for (i = 0; i < 256; ++i) {

        if (sregnames[i]) {

            cpu_SR[i] = tcg_global_mem_new_i32(TCG_AREG0,

                    offsetof(CPUXtensaState, sregs[i]),

                    sregnames[i]);

        }

    }



    for (i = 0; i < 256; ++i) {

        if (uregnames[i]) {

            cpu_UR[i] = tcg_global_mem_new_i32(TCG_AREG0,

                    offsetof(CPUXtensaState, uregs[i]),

                    uregnames[i]);

        }

    }

#define GEN_HELPER 2

#include "helper.h"

}
