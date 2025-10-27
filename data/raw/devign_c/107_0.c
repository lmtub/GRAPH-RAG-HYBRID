void HELPER(ucf64_cmps)(float32 a, float32 b, uint32_t c, CPUUniCore32State *env)

{

    int flag;

    flag = float32_compare_quiet(a, b, &env->ucf64.fp_status);

    env->CF = 0;

    switch (c & 0x7) {

    case 0: /* F */

        break;

    case 1: /* UN */

        if (flag == 2) {

            env->CF = 1;

        }

        break;

    case 2: /* EQ */

        if (flag == 0) {

            env->CF = 1;

        }

        break;

    case 3: /* UEQ */

        if ((flag == 0) || (flag == 2)) {

            env->CF = 1;

        }

        break;

    case 4: /* OLT */

        if (flag == -1) {

            env->CF = 1;

        }

        break;

    case 5: /* ULT */

        if ((flag == -1) || (flag == 2)) {

            env->CF = 1;

        }

        break;

    case 6: /* OLE */

        if ((flag == -1) || (flag == 0)) {

            env->CF = 1;

        }

        break;

    case 7: /* ULE */

        if (flag != 1) {

            env->CF = 1;

        }

        break;

    }

    env->ucf64.xregs[UC32_UCF64_FPSCR] = (env->CF << 29)

                    | (env->ucf64.xregs[UC32_UCF64_FPSCR] & 0x0fffffff);

}
