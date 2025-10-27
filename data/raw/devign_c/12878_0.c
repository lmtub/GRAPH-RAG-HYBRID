static int cpu_gdb_read_register(CPUState *env, uint8_t *mem_buf, int n)

{

    if (n < 32) {

        GET_REGL(env->active_tc.gpr[n]);

    }

    if (env->CP0_Config1 & (1 << CP0C1_FP)) {

        if (n >= 38 && n < 70) {

            if (env->CP0_Status & (1 << CP0St_FR))

		GET_REGL(env->active_fpu.fpr[n - 38].d);

            else

		GET_REGL(env->active_fpu.fpr[n - 38].w[FP_ENDIAN_IDX]);

        }

        switch (n) {

        case 70: GET_REGL((int32_t)env->active_fpu.fcr31);

        case 71: GET_REGL((int32_t)env->active_fpu.fcr0);

        }

    }

    switch (n) {

    case 32: GET_REGL((int32_t)env->CP0_Status);

    case 33: GET_REGL(env->active_tc.LO[0]);

    case 34: GET_REGL(env->active_tc.HI[0]);

    case 35: GET_REGL(env->CP0_BadVAddr);

    case 36: GET_REGL((int32_t)env->CP0_Cause);

    case 37: GET_REGL(env->active_tc.PC);

    case 72: GET_REGL(0); /* fp */

    case 89: GET_REGL((int32_t)env->CP0_PRid);

    }

    if (n >= 73 && n <= 88) {

	/* 16 embedded regs.  */

	GET_REGL(0);

    }



    return 0;

}
