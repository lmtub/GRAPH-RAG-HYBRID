static int disas_vfp_insn(CPUARMState * env, DisasContext *s, uint32_t insn)

{

    uint32_t rd, rn, rm, op, i, n, offset, delta_d, delta_m, bank_mask;

    int dp, veclen;

    TCGv addr;

    TCGv tmp;

    TCGv tmp2;



    if (!arm_feature(env, ARM_FEATURE_VFP))

        return 1;



    if (!s->vfp_enabled) {

        /* VFP disabled.  Only allow fmxr/fmrx to/from some control regs.  */

        if ((insn & 0x0fe00fff) != 0x0ee00a10)

            return 1;

        rn = (insn >> 16) & 0xf;

        if (rn != ARM_VFP_FPSID && rn != ARM_VFP_FPEXC

            && rn != ARM_VFP_MVFR1 && rn != ARM_VFP_MVFR0)

            return 1;

    }

    dp = ((insn & 0xf00) == 0xb00);

    switch ((insn >> 24) & 0xf) {

    case 0xe:

        if (insn & (1 << 4)) {

            /* single register transfer */

            rd = (insn >> 12) & 0xf;

            if (dp) {

                int size;

                int pass;



                VFP_DREG_N(rn, insn);

                if (insn & 0xf)

                    return 1;

                if (insn & 0x00c00060

                    && !arm_feature(env, ARM_FEATURE_NEON))

                    return 1;



                pass = (insn >> 21) & 1;

                if (insn & (1 << 22)) {

                    size = 0;

                    offset = ((insn >> 5) & 3) * 8;

                } else if (insn & (1 << 5)) {

                    size = 1;

                    offset = (insn & (1 << 6)) ? 16 : 0;

                } else {

                    size = 2;

                    offset = 0;

                }

                if (insn & ARM_CP_RW_BIT) {

                    /* vfp->arm */

                    tmp = neon_load_reg(rn, pass);

                    switch (size) {

                    case 0:

                        if (offset)

                            tcg_gen_shri_i32(tmp, tmp, offset);

                        if (insn & (1 << 23))

                            gen_uxtb(tmp);

                        else

                            gen_sxtb(tmp);

                        break;

                    case 1:

                        if (insn & (1 << 23)) {

                            if (offset) {

                                tcg_gen_shri_i32(tmp, tmp, 16);

                            } else {

                                gen_uxth(tmp);

                            }

                        } else {

                            if (offset) {

                                tcg_gen_sari_i32(tmp, tmp, 16);

                            } else {

                                gen_sxth(tmp);

                            }

                        }

                        break;

                    case 2:

                        break;

                    }

                    store_reg(s, rd, tmp);

                } else {

                    /* arm->vfp */

                    tmp = load_reg(s, rd);

                    if (insn & (1 << 23)) {

                        /* VDUP */

                        if (size == 0) {

                            gen_neon_dup_u8(tmp, 0);

                        } else if (size == 1) {

                            gen_neon_dup_low16(tmp);

                        }

                        for (n = 0; n <= pass * 2; n++) {

                            tmp2 = tcg_temp_new_i32();

                            tcg_gen_mov_i32(tmp2, tmp);

                            neon_store_reg(rn, n, tmp2);

                        }

                        neon_store_reg(rn, n, tmp);

                    } else {

                        /* VMOV */

                        switch (size) {

                        case 0:

                            tmp2 = neon_load_reg(rn, pass);

                            tcg_gen_deposit_i32(tmp, tmp2, tmp, offset, 8);

                            tcg_temp_free_i32(tmp2);

                            break;

                        case 1:

                            tmp2 = neon_load_reg(rn, pass);

                            tcg_gen_deposit_i32(tmp, tmp2, tmp, offset, 16);

                            tcg_temp_free_i32(tmp2);

                            break;

                        case 2:

                            break;

                        }

                        neon_store_reg(rn, pass, tmp);

                    }

                }

            } else { /* !dp */

                if ((insn & 0x6f) != 0x00)

                    return 1;

                rn = VFP_SREG_N(insn);

                if (insn & ARM_CP_RW_BIT) {

                    /* vfp->arm */

                    if (insn & (1 << 21)) {

                        /* system register */

                        rn >>= 1;



                        switch (rn) {

                        case ARM_VFP_FPSID:

                            /* VFP2 allows access to FSID from userspace.

                               VFP3 restricts all id registers to privileged

                               accesses.  */

                            if (IS_USER(s)

                                && arm_feature(env, ARM_FEATURE_VFP3))

                                return 1;

                            tmp = load_cpu_field(vfp.xregs[rn]);

                            break;

                        case ARM_VFP_FPEXC:

                            if (IS_USER(s))

                                return 1;

                            tmp = load_cpu_field(vfp.xregs[rn]);

                            break;

                        case ARM_VFP_FPINST:

                        case ARM_VFP_FPINST2:

                            /* Not present in VFP3.  */

                            if (IS_USER(s)

                                || arm_feature(env, ARM_FEATURE_VFP3))

                                return 1;

                            tmp = load_cpu_field(vfp.xregs[rn]);

                            break;

                        case ARM_VFP_FPSCR:

                            if (rd == 15) {

                                tmp = load_cpu_field(vfp.xregs[ARM_VFP_FPSCR]);

                                tcg_gen_andi_i32(tmp, tmp, 0xf0000000);

                            } else {

                                tmp = tcg_temp_new_i32();

                                gen_helper_vfp_get_fpscr(tmp, cpu_env);

                            }

                            break;

                        case ARM_VFP_MVFR0:

                        case ARM_VFP_MVFR1:

                            if (IS_USER(s)

                                || !arm_feature(env, ARM_FEATURE_MVFR))

                                return 1;

                            tmp = load_cpu_field(vfp.xregs[rn]);

                            break;

                        default:

                            return 1;

                        }

                    } else {

                        gen_mov_F0_vreg(0, rn);

                        tmp = gen_vfp_mrs();

                    }

                    if (rd == 15) {

                        /* Set the 4 flag bits in the CPSR.  */

                        gen_set_nzcv(tmp);

                        tcg_temp_free_i32(tmp);

                    } else {

                        store_reg(s, rd, tmp);

                    }

                } else {

                    /* arm->vfp */

                    tmp = load_reg(s, rd);

                    if (insn & (1 << 21)) {

                        rn >>= 1;

                        /* system register */

                        switch (rn) {

                        case ARM_VFP_FPSID:

                        case ARM_VFP_MVFR0:

                        case ARM_VFP_MVFR1:

                            /* Writes are ignored.  */

                            break;

                        case ARM_VFP_FPSCR:

                            gen_helper_vfp_set_fpscr(cpu_env, tmp);

                            tcg_temp_free_i32(tmp);

                            gen_lookup_tb(s);

                            break;

                        case ARM_VFP_FPEXC:

                            if (IS_USER(s))

                                return 1;

                            /* TODO: VFP subarchitecture support.

                             * For now, keep the EN bit only */

                            tcg_gen_andi_i32(tmp, tmp, 1 << 30);

                            store_cpu_field(tmp, vfp.xregs[rn]);

                            gen_lookup_tb(s);

                            break;

                        case ARM_VFP_FPINST:

                        case ARM_VFP_FPINST2:

                            store_cpu_field(tmp, vfp.xregs[rn]);

                            break;

                        default:

                            return 1;

                        }

                    } else {

                        gen_vfp_msr(tmp);

                        gen_mov_vreg_F0(0, rn);

                    }

                }

            }

        } else {

            /* data processing */

            /* The opcode is in bits 23, 21, 20 and 6.  */

            op = ((insn >> 20) & 8) | ((insn >> 19) & 6) | ((insn >> 6) & 1);

            if (dp) {

                if (op == 15) {

                    /* rn is opcode */

                    rn = ((insn >> 15) & 0x1e) | ((insn >> 7) & 1);

                } else {

                    /* rn is register number */

                    VFP_DREG_N(rn, insn);

                }



                if (op == 15 && (rn == 15 || ((rn & 0x1c) == 0x18))) {

                    /* Integer or single precision destination.  */

                    rd = VFP_SREG_D(insn);

                } else {

                    VFP_DREG_D(rd, insn);

                }

                if (op == 15 &&

                    (((rn & 0x1c) == 0x10) || ((rn & 0x14) == 0x14))) {

                    /* VCVT from int is always from S reg regardless of dp bit.

                     * VCVT with immediate frac_bits has same format as SREG_M

                     */

                    rm = VFP_SREG_M(insn);

                } else {

                    VFP_DREG_M(rm, insn);

                }

            } else {

                rn = VFP_SREG_N(insn);

                if (op == 15 && rn == 15) {

                    /* Double precision destination.  */

                    VFP_DREG_D(rd, insn);

                } else {

                    rd = VFP_SREG_D(insn);

                }

                /* NB that we implicitly rely on the encoding for the frac_bits

                 * in VCVT of fixed to float being the same as that of an SREG_M

                 */

                rm = VFP_SREG_M(insn);

            }



            veclen = s->vec_len;

            if (op == 15 && rn > 3)

                veclen = 0;



            /* Shut up compiler warnings.  */

            delta_m = 0;

            delta_d = 0;

            bank_mask = 0;



            if (veclen > 0) {

                if (dp)

                    bank_mask = 0xc;

                else

                    bank_mask = 0x18;



                /* Figure out what type of vector operation this is.  */

                if ((rd & bank_mask) == 0) {

                    /* scalar */

                    veclen = 0;

                } else {

                    if (dp)

                        delta_d = (s->vec_stride >> 1) + 1;

                    else

                        delta_d = s->vec_stride + 1;



                    if ((rm & bank_mask) == 0) {

                        /* mixed scalar/vector */

                        delta_m = 0;

                    } else {

                        /* vector */

                        delta_m = delta_d;

                    }

                }

            }



            /* Load the initial operands.  */

            if (op == 15) {

                switch (rn) {

                case 16:

                case 17:

                    /* Integer source */

                    gen_mov_F0_vreg(0, rm);

                    break;

                case 8:

                case 9:

                    /* Compare */

                    gen_mov_F0_vreg(dp, rd);

                    gen_mov_F1_vreg(dp, rm);

                    break;

                case 10:

                case 11:

                    /* Compare with zero */

                    gen_mov_F0_vreg(dp, rd);

                    gen_vfp_F1_ld0(dp);

                    break;

                case 20:

                case 21:

                case 22:

                case 23:

                case 28:

                case 29:

                case 30:

                case 31:

                    /* Source and destination the same.  */

                    gen_mov_F0_vreg(dp, rd);

                    break;

                case 4:

                case 5:

                case 6:

                case 7:

                    /* VCVTB, VCVTT: only present with the halfprec extension,

                     * UNPREDICTABLE if bit 8 is set (we choose to UNDEF)

                     */

                    if (dp || !arm_feature(env, ARM_FEATURE_VFP_FP16)) {

                        return 1;

                    }

                    /* Otherwise fall through */

                default:

                    /* One source operand.  */

                    gen_mov_F0_vreg(dp, rm);

                    break;

                }

            } else {

                /* Two source operands.  */

                gen_mov_F0_vreg(dp, rn);

                gen_mov_F1_vreg(dp, rm);

            }



            for (;;) {

                /* Perform the calculation.  */

                switch (op) {

                case 0: /* VMLA: fd + (fn * fm) */

                    /* Note that order of inputs to the add matters for NaNs */

                    gen_vfp_F1_mul(dp);

                    gen_mov_F0_vreg(dp, rd);

                    gen_vfp_add(dp);

                    break;

                case 1: /* VMLS: fd + -(fn * fm) */

                    gen_vfp_mul(dp);

                    gen_vfp_F1_neg(dp);

                    gen_mov_F0_vreg(dp, rd);

                    gen_vfp_add(dp);

                    break;

                case 2: /* VNMLS: -fd + (fn * fm) */

                    /* Note that it isn't valid to replace (-A + B) with (B - A)

                     * or similar plausible looking simplifications

                     * because this will give wrong results for NaNs.

                     */

                    gen_vfp_F1_mul(dp);

                    gen_mov_F0_vreg(dp, rd);

                    gen_vfp_neg(dp);

                    gen_vfp_add(dp);

                    break;

                case 3: /* VNMLA: -fd + -(fn * fm) */

                    gen_vfp_mul(dp);

                    gen_vfp_F1_neg(dp);

                    gen_mov_F0_vreg(dp, rd);

                    gen_vfp_neg(dp);

                    gen_vfp_add(dp);

                    break;

                case 4: /* mul: fn * fm */

                    gen_vfp_mul(dp);

                    break;

                case 5: /* nmul: -(fn * fm) */

                    gen_vfp_mul(dp);

                    gen_vfp_neg(dp);

                    break;

                case 6: /* add: fn + fm */

                    gen_vfp_add(dp);

                    break;

                case 7: /* sub: fn - fm */

                    gen_vfp_sub(dp);

                    break;

                case 8: /* div: fn / fm */

                    gen_vfp_div(dp);

                    break;

                case 10: /* VFNMA : fd = muladd(-fd,  fn, fm) */

                case 11: /* VFNMS : fd = muladd(-fd, -fn, fm) */

                case 12: /* VFMA  : fd = muladd( fd,  fn, fm) */

                case 13: /* VFMS  : fd = muladd( fd, -fn, fm) */

                    /* These are fused multiply-add, and must be done as one

                     * floating point operation with no rounding between the

                     * multiplication and addition steps.

                     * NB that doing the negations here as separate steps is

                     * correct : an input NaN should come out with its sign bit

                     * flipped if it is a negated-input.

                     */

                    if (!arm_feature(env, ARM_FEATURE_VFP4)) {

                        return 1;

                    }

                    if (dp) {

                        TCGv_ptr fpst;

                        TCGv_i64 frd;

                        if (op & 1) {

                            /* VFNMS, VFMS */

                            gen_helper_vfp_negd(cpu_F0d, cpu_F0d);

                        }

                        frd = tcg_temp_new_i64();

                        tcg_gen_ld_f64(frd, cpu_env, vfp_reg_offset(dp, rd));

                        if (op & 2) {

                            /* VFNMA, VFNMS */

                            gen_helper_vfp_negd(frd, frd);

                        }

                        fpst = get_fpstatus_ptr(0);

                        gen_helper_vfp_muladdd(cpu_F0d, cpu_F0d,

                                               cpu_F1d, frd, fpst);

                        tcg_temp_free_ptr(fpst);

                        tcg_temp_free_i64(frd);

                    } else {

                        TCGv_ptr fpst;

                        TCGv_i32 frd;

                        if (op & 1) {

                            /* VFNMS, VFMS */

                            gen_helper_vfp_negs(cpu_F0s, cpu_F0s);

                        }

                        frd = tcg_temp_new_i32();

                        tcg_gen_ld_f32(frd, cpu_env, vfp_reg_offset(dp, rd));

                        if (op & 2) {

                            gen_helper_vfp_negs(frd, frd);

                        }

                        fpst = get_fpstatus_ptr(0);

                        gen_helper_vfp_muladds(cpu_F0s, cpu_F0s,

                                               cpu_F1s, frd, fpst);

                        tcg_temp_free_ptr(fpst);

                        tcg_temp_free_i32(frd);

                    }

                    break;

                case 14: /* fconst */

                    if (!arm_feature(env, ARM_FEATURE_VFP3))

                      return 1;



                    n = (insn << 12) & 0x80000000;

                    i = ((insn >> 12) & 0x70) | (insn & 0xf);

                    if (dp) {

                        if (i & 0x40)

                            i |= 0x3f80;

                        else

                            i |= 0x4000;

                        n |= i << 16;

                        tcg_gen_movi_i64(cpu_F0d, ((uint64_t)n) << 32);

                    } else {

                        if (i & 0x40)

                            i |= 0x780;

                        else

                            i |= 0x800;

                        n |= i << 19;

                        tcg_gen_movi_i32(cpu_F0s, n);

                    }

                    break;

                case 15: /* extension space */

                    switch (rn) {

                    case 0: /* cpy */

                        /* no-op */

                        break;

                    case 1: /* abs */

                        gen_vfp_abs(dp);

                        break;

                    case 2: /* neg */

                        gen_vfp_neg(dp);

                        break;

                    case 3: /* sqrt */

                        gen_vfp_sqrt(dp);

                        break;

                    case 4: /* vcvtb.f32.f16 */

                        tmp = gen_vfp_mrs();

                        tcg_gen_ext16u_i32(tmp, tmp);

                        gen_helper_vfp_fcvt_f16_to_f32(cpu_F0s, tmp, cpu_env);

                        tcg_temp_free_i32(tmp);

                        break;

                    case 5: /* vcvtt.f32.f16 */

                        tmp = gen_vfp_mrs();

                        tcg_gen_shri_i32(tmp, tmp, 16);

                        gen_helper_vfp_fcvt_f16_to_f32(cpu_F0s, tmp, cpu_env);

                        tcg_temp_free_i32(tmp);

                        break;

                    case 6: /* vcvtb.f16.f32 */

                        tmp = tcg_temp_new_i32();

                        gen_helper_vfp_fcvt_f32_to_f16(tmp, cpu_F0s, cpu_env);

                        gen_mov_F0_vreg(0, rd);

                        tmp2 = gen_vfp_mrs();

                        tcg_gen_andi_i32(tmp2, tmp2, 0xffff0000);

                        tcg_gen_or_i32(tmp, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        gen_vfp_msr(tmp);

                        break;

                    case 7: /* vcvtt.f16.f32 */

                        tmp = tcg_temp_new_i32();

                        gen_helper_vfp_fcvt_f32_to_f16(tmp, cpu_F0s, cpu_env);

                        tcg_gen_shli_i32(tmp, tmp, 16);

                        gen_mov_F0_vreg(0, rd);

                        tmp2 = gen_vfp_mrs();

                        tcg_gen_ext16u_i32(tmp2, tmp2);

                        tcg_gen_or_i32(tmp, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        gen_vfp_msr(tmp);

                        break;

                    case 8: /* cmp */

                        gen_vfp_cmp(dp);

                        break;

                    case 9: /* cmpe */

                        gen_vfp_cmpe(dp);

                        break;

                    case 10: /* cmpz */

                        gen_vfp_cmp(dp);

                        break;

                    case 11: /* cmpez */

                        gen_vfp_F1_ld0(dp);

                        gen_vfp_cmpe(dp);

                        break;

                    case 15: /* single<->double conversion */

                        if (dp)

                            gen_helper_vfp_fcvtsd(cpu_F0s, cpu_F0d, cpu_env);

                        else

                            gen_helper_vfp_fcvtds(cpu_F0d, cpu_F0s, cpu_env);

                        break;

                    case 16: /* fuito */

                        gen_vfp_uito(dp, 0);

                        break;

                    case 17: /* fsito */

                        gen_vfp_sito(dp, 0);

                        break;

                    case 20: /* fshto */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_shto(dp, 16 - rm, 0);

                        break;

                    case 21: /* fslto */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_slto(dp, 32 - rm, 0);

                        break;

                    case 22: /* fuhto */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_uhto(dp, 16 - rm, 0);

                        break;

                    case 23: /* fulto */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_ulto(dp, 32 - rm, 0);

                        break;

                    case 24: /* ftoui */

                        gen_vfp_toui(dp, 0);

                        break;

                    case 25: /* ftouiz */

                        gen_vfp_touiz(dp, 0);

                        break;

                    case 26: /* ftosi */

                        gen_vfp_tosi(dp, 0);

                        break;

                    case 27: /* ftosiz */

                        gen_vfp_tosiz(dp, 0);

                        break;

                    case 28: /* ftosh */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_tosh(dp, 16 - rm, 0);

                        break;

                    case 29: /* ftosl */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_tosl(dp, 32 - rm, 0);

                        break;

                    case 30: /* ftouh */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_touh(dp, 16 - rm, 0);

                        break;

                    case 31: /* ftoul */

                        if (!arm_feature(env, ARM_FEATURE_VFP3))

                          return 1;

                        gen_vfp_toul(dp, 32 - rm, 0);

                        break;

                    default: /* undefined */

                        return 1;

                    }

                    break;

                default: /* undefined */

                    return 1;

                }



                /* Write back the result.  */

                if (op == 15 && (rn >= 8 && rn <= 11))

                    ; /* Comparison, do nothing.  */

                else if (op == 15 && dp && ((rn & 0x1c) == 0x18))

                    /* VCVT double to int: always integer result. */

                    gen_mov_vreg_F0(0, rd);

                else if (op == 15 && rn == 15)

                    /* conversion */

                    gen_mov_vreg_F0(!dp, rd);

                else

                    gen_mov_vreg_F0(dp, rd);



                /* break out of the loop if we have finished  */

                if (veclen == 0)

                    break;



                if (op == 15 && delta_m == 0) {

                    /* single source one-many */

                    while (veclen--) {

                        rd = ((rd + delta_d) & (bank_mask - 1))

                             | (rd & bank_mask);

                        gen_mov_vreg_F0(dp, rd);

                    }

                    break;

                }

                /* Setup the next operands.  */

                veclen--;

                rd = ((rd + delta_d) & (bank_mask - 1))

                     | (rd & bank_mask);



                if (op == 15) {

                    /* One source operand.  */

                    rm = ((rm + delta_m) & (bank_mask - 1))

                         | (rm & bank_mask);

                    gen_mov_F0_vreg(dp, rm);

                } else {

                    /* Two source operands.  */

                    rn = ((rn + delta_d) & (bank_mask - 1))

                         | (rn & bank_mask);

                    gen_mov_F0_vreg(dp, rn);

                    if (delta_m) {

                        rm = ((rm + delta_m) & (bank_mask - 1))

                             | (rm & bank_mask);

                        gen_mov_F1_vreg(dp, rm);

                    }

                }

            }

        }

        break;

    case 0xc:

    case 0xd:

        if ((insn & 0x03e00000) == 0x00400000) {

            /* two-register transfer */

            rn = (insn >> 16) & 0xf;

            rd = (insn >> 12) & 0xf;

            if (dp) {

                VFP_DREG_M(rm, insn);

            } else {

                rm = VFP_SREG_M(insn);

            }



            if (insn & ARM_CP_RW_BIT) {

                /* vfp->arm */

                if (dp) {

                    gen_mov_F0_vreg(0, rm * 2);

                    tmp = gen_vfp_mrs();

                    store_reg(s, rd, tmp);

                    gen_mov_F0_vreg(0, rm * 2 + 1);

                    tmp = gen_vfp_mrs();

                    store_reg(s, rn, tmp);

                } else {

                    gen_mov_F0_vreg(0, rm);

                    tmp = gen_vfp_mrs();

                    store_reg(s, rd, tmp);

                    gen_mov_F0_vreg(0, rm + 1);

                    tmp = gen_vfp_mrs();

                    store_reg(s, rn, tmp);

                }

            } else {

                /* arm->vfp */

                if (dp) {

                    tmp = load_reg(s, rd);

                    gen_vfp_msr(tmp);

                    gen_mov_vreg_F0(0, rm * 2);

                    tmp = load_reg(s, rn);

                    gen_vfp_msr(tmp);

                    gen_mov_vreg_F0(0, rm * 2 + 1);

                } else {

                    tmp = load_reg(s, rd);

                    gen_vfp_msr(tmp);

                    gen_mov_vreg_F0(0, rm);

                    tmp = load_reg(s, rn);

                    gen_vfp_msr(tmp);

                    gen_mov_vreg_F0(0, rm + 1);

                }

            }

        } else {

            /* Load/store */

            rn = (insn >> 16) & 0xf;

            if (dp)

                VFP_DREG_D(rd, insn);

            else

                rd = VFP_SREG_D(insn);

            if ((insn & 0x01200000) == 0x01000000) {

                /* Single load/store */

                offset = (insn & 0xff) << 2;

                if ((insn & (1 << 23)) == 0)

                    offset = -offset;

                if (s->thumb && rn == 15) {

                    /* This is actually UNPREDICTABLE */

                    addr = tcg_temp_new_i32();

                    tcg_gen_movi_i32(addr, s->pc & ~2);

                } else {

                    addr = load_reg(s, rn);

                }

                tcg_gen_addi_i32(addr, addr, offset);

                if (insn & (1 << 20)) {

                    gen_vfp_ld(s, dp, addr);

                    gen_mov_vreg_F0(dp, rd);

                } else {

                    gen_mov_F0_vreg(dp, rd);

                    gen_vfp_st(s, dp, addr);

                }

                tcg_temp_free_i32(addr);

            } else {

                /* load/store multiple */

                int w = insn & (1 << 21);

                if (dp)

                    n = (insn >> 1) & 0x7f;

                else

                    n = insn & 0xff;



                if (w && !(((insn >> 23) ^ (insn >> 24)) & 1)) {

                    /* P == U , W == 1  => UNDEF */

                    return 1;

                }

                if (n == 0 || (rd + n) > 32 || (dp && n > 16)) {

                    /* UNPREDICTABLE cases for bad immediates: we choose to

                     * UNDEF to avoid generating huge numbers of TCG ops

                     */

                    return 1;

                }

                if (rn == 15 && w) {

                    /* writeback to PC is UNPREDICTABLE, we choose to UNDEF */

                    return 1;

                }



                if (s->thumb && rn == 15) {

                    /* This is actually UNPREDICTABLE */

                    addr = tcg_temp_new_i32();

                    tcg_gen_movi_i32(addr, s->pc & ~2);

                } else {

                    addr = load_reg(s, rn);

                }

                if (insn & (1 << 24)) /* pre-decrement */

                    tcg_gen_addi_i32(addr, addr, -((insn & 0xff) << 2));



                if (dp)

                    offset = 8;

                else

                    offset = 4;

                for (i = 0; i < n; i++) {

                    if (insn & ARM_CP_RW_BIT) {

                        /* load */

                        gen_vfp_ld(s, dp, addr);

                        gen_mov_vreg_F0(dp, rd + i);

                    } else {

                        /* store */

                        gen_mov_F0_vreg(dp, rd + i);

                        gen_vfp_st(s, dp, addr);

                    }

                    tcg_gen_addi_i32(addr, addr, offset);

                }

                if (w) {

                    /* writeback */

                    if (insn & (1 << 24))

                        offset = -offset * n;

                    else if (dp && (insn & 1))

                        offset = 4;

                    else

                        offset = 0;



                    if (offset != 0)

                        tcg_gen_addi_i32(addr, addr, offset);

                    store_reg(s, rn, addr);

                } else {

                    tcg_temp_free_i32(addr);

                }

            }

        }

        break;

    default:

        /* Should never happen.  */

        return 1;

    }

    return 0;

}
