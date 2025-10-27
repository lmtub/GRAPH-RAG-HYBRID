static unsigned int dec10_quick_imm(DisasContext *dc)

{

    int32_t imm, simm;

    int op;



    /* sign extend.  */

    imm = dc->ir & ((1 << 6) - 1);

    simm = (int8_t) (imm << 2);

    simm >>= 2;

    switch (dc->opcode) {

        case CRISV10_QIMM_BDAP_R0:

        case CRISV10_QIMM_BDAP_R1:

        case CRISV10_QIMM_BDAP_R2:

        case CRISV10_QIMM_BDAP_R3:

            simm = (int8_t)dc->ir;

            LOG_DIS("bdap %d $r%d\n", simm, dc->dst);

            LOG_DIS("pc=%x mode=%x quickimm %d r%d r%d\n",

                     dc->pc, dc->mode, dc->opcode, dc->src, dc->dst);

            cris_set_prefix(dc);

            if (dc->dst == 15) {

                tcg_gen_movi_tl(cpu_PR[PR_PREFIX], dc->pc + 2 + simm);

            } else {

                tcg_gen_addi_tl(cpu_PR[PR_PREFIX], cpu_R[dc->dst], simm);

            }

            break;



        case CRISV10_QIMM_MOVEQ:

            LOG_DIS("moveq %d, $r%d\n", simm, dc->dst);



            cris_cc_mask(dc, CC_MASK_NZVC);

            cris_alu(dc, CC_OP_MOVE, cpu_R[dc->dst],

                     cpu_R[dc->dst], tcg_const_tl(simm), 4);

            break;

        case CRISV10_QIMM_CMPQ:

            LOG_DIS("cmpq %d, $r%d\n", simm, dc->dst);



            cris_cc_mask(dc, CC_MASK_NZVC);

            cris_alu(dc, CC_OP_CMP, cpu_R[dc->dst],

                     cpu_R[dc->dst], tcg_const_tl(simm), 4);

            break;

        case CRISV10_QIMM_ADDQ:

            LOG_DIS("addq %d, $r%d\n", imm, dc->dst);



            cris_cc_mask(dc, CC_MASK_NZVC);

            cris_alu(dc, CC_OP_ADD, cpu_R[dc->dst],

                     cpu_R[dc->dst], tcg_const_tl(imm), 4);

            break;

        case CRISV10_QIMM_ANDQ:

            LOG_DIS("andq %d, $r%d\n", simm, dc->dst);



            cris_cc_mask(dc, CC_MASK_NZVC);

            cris_alu(dc, CC_OP_AND, cpu_R[dc->dst],

                     cpu_R[dc->dst], tcg_const_tl(simm), 4);

            break;

        case CRISV10_QIMM_ASHQ:

            LOG_DIS("ashq %d, $r%d\n", simm, dc->dst);



            cris_cc_mask(dc, CC_MASK_NZVC);

            op = imm & (1 << 5);

            imm &= 0x1f;

            if (op) {

                cris_alu(dc, CC_OP_ASR, cpu_R[dc->dst],

                          cpu_R[dc->dst], tcg_const_tl(imm), 4);

            } else {

                /* BTST */

                cris_update_cc_op(dc, CC_OP_FLAGS, 4);

                gen_helper_btst(cpu_PR[PR_CCS], cpu_R[dc->dst],

                           tcg_const_tl(imm), cpu_PR[PR_CCS]);

            }

            break;

        case CRISV10_QIMM_LSHQ:

            LOG_DIS("lshq %d, $r%d\n", simm, dc->dst);



            op = CC_OP_LSL;

            if (imm & (1 << 5)) {

                op = CC_OP_LSR; 

            }

            imm &= 0x1f;

            cris_cc_mask(dc, CC_MASK_NZVC);

            cris_alu(dc, op, cpu_R[dc->dst],

                     cpu_R[dc->dst], tcg_const_tl(imm), 4);

            break;

        case CRISV10_QIMM_SUBQ:

            LOG_DIS("subq %d, $r%d\n", imm, dc->dst);



            cris_cc_mask(dc, CC_MASK_NZVC);

            cris_alu(dc, CC_OP_SUB, cpu_R[dc->dst],

                     cpu_R[dc->dst], tcg_const_tl(imm), 4);

            break;

        case CRISV10_QIMM_ORQ:

            LOG_DIS("andq %d, $r%d\n", simm, dc->dst);



            cris_cc_mask(dc, CC_MASK_NZVC);

            cris_alu(dc, CC_OP_OR, cpu_R[dc->dst],

                     cpu_R[dc->dst], tcg_const_tl(simm), 4);

            break;



        case CRISV10_QIMM_BCC_R0:

            if (!dc->ir) {

                cpu_abort(dc->env, "opcode zero\n");

            }

        case CRISV10_QIMM_BCC_R1:

        case CRISV10_QIMM_BCC_R2:

        case CRISV10_QIMM_BCC_R3:

            imm = dc->ir & 0xff;

            /* bit 0 is a sign bit.  */

            if (imm & 1) {

                imm |= 0xffffff00;   /* sign extend.  */

                imm &= ~1;           /* get rid of the sign bit.  */

            }

            imm += 2;

            LOG_DIS("b%s %d\n", cc_name(dc->cond), imm);



            cris_cc_mask(dc, 0);

            cris_prepare_cc_branch(dc, imm, dc->cond); 

            break;



        default:

            LOG_DIS("pc=%x mode=%x quickimm %d r%d r%d\n",

                     dc->pc, dc->mode, dc->opcode, dc->src, dc->dst);

            cpu_abort(dc->env, "Unhandled quickimm\n");

            break;

    }

    return 2;

}
