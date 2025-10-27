DISAS_INSN(cas2l)

{

    uint16_t ext1, ext2;

    TCGv addr1, addr2, regs;



    /* cas2 Dc1:Dc2,Du1:Du2,(Rn1):(Rn2) */



    ext1 = read_im16(env, s);



    if (ext1 & 0x8000) {

        /* Address Register */

        addr1 = AREG(ext1, 12);

    } else {

        /* Data Register */

        addr1 = DREG(ext1, 12);

    }



    ext2 = read_im16(env, s);

    if (ext2 & 0x8000) {

        /* Address Register */

        addr2 = AREG(ext2, 12);

    } else {

        /* Data Register */

        addr2 = DREG(ext2, 12);

    }



    /* if (R1) == Dc1 && (R2) == Dc2 then

     *     (R1) = Du1

     *     (R2) = Du2

     * else

     *     Dc1 = (R1)

     *     Dc2 = (R2)

     */



    regs = tcg_const_i32(REG(ext2, 6) |

                         (REG(ext1, 6) << 3) |

                         (REG(ext2, 0) << 6) |

                         (REG(ext1, 0) << 9));

    gen_helper_cas2l(cpu_env, regs, addr1, addr2);

    tcg_temp_free(regs);



    /* Note that cas2l also assigned to env->cc_op.  */

    s->cc_op = CC_OP_CMPL;

    s->cc_op_synced = 1;

}
