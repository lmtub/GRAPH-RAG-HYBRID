void helper_ldda_asi(target_ulong addr, int asi, int rd)

{

    if ((asi < 0x80 && (env->pstate & PS_PRIV) == 0)

        || ((env->def->features & CPU_FEATURE_HYPV)

            && asi >= 0x30 && asi < 0x80

            && !(env->hpstate & HS_PRIV)))

        raise_exception(TT_PRIV_ACT);



    switch (asi) {

    case 0x24: // Nucleus quad LDD 128 bit atomic

    case 0x2c: // Nucleus quad LDD 128 bit atomic LE

        helper_check_align(addr, 0xf);

        if (rd == 0) {

            env->gregs[1] = ldq_kernel(addr + 8);

            if (asi == 0x2c)

                bswap64s(&env->gregs[1]);

        } else if (rd < 8) {

            env->gregs[rd] = ldq_kernel(addr);

            env->gregs[rd + 1] = ldq_kernel(addr + 8);

            if (asi == 0x2c) {

                bswap64s(&env->gregs[rd]);

                bswap64s(&env->gregs[rd + 1]);

            }

        } else {

            env->regwptr[rd] = ldq_kernel(addr);

            env->regwptr[rd + 1] = ldq_kernel(addr + 8);

            if (asi == 0x2c) {

                bswap64s(&env->regwptr[rd]);

                bswap64s(&env->regwptr[rd + 1]);

            }

        }

        break;

    default:

        helper_check_align(addr, 0x3);

        if (rd == 0)

            env->gregs[1] = helper_ld_asi(addr + 4, asi, 4, 0);

        else if (rd < 8) {

            env->gregs[rd] = helper_ld_asi(addr, asi, 4, 0);

            env->gregs[rd + 1] = helper_ld_asi(addr + 4, asi, 4, 0);

        } else {

            env->regwptr[rd] = helper_ld_asi(addr, asi, 4, 0);

            env->regwptr[rd + 1] = helper_ld_asi(addr + 4, asi, 4, 0);

        }

        break;

    }

}
