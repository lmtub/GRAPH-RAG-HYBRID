static void dec_wcsr(DisasContext *dc)

{

    int no;



    LOG_DIS("wcsr r%d, %d\n", dc->r1, dc->csr);



    switch (dc->csr) {

    case CSR_IE:

        tcg_gen_mov_tl(cpu_ie, cpu_R[dc->r1]);

        tcg_gen_movi_tl(cpu_pc, dc->pc + 4);

        dc->is_jmp = DISAS_UPDATE;

        break;

    case CSR_IM:

        /* mark as an io operation because it could cause an interrupt */

        if (use_icount) {

            gen_io_start();

        }

        gen_helper_wcsr_im(cpu_env, cpu_R[dc->r1]);

        tcg_gen_movi_tl(cpu_pc, dc->pc + 4);

        if (use_icount) {

            gen_io_end();

        }

        dc->is_jmp = DISAS_UPDATE;

        break;

    case CSR_IP:

        /* mark as an io operation because it could cause an interrupt */

        if (use_icount) {

            gen_io_start();

        }

        gen_helper_wcsr_ip(cpu_env, cpu_R[dc->r1]);

        tcg_gen_movi_tl(cpu_pc, dc->pc + 4);

        if (use_icount) {

            gen_io_end();

        }

        dc->is_jmp = DISAS_UPDATE;

        break;

    case CSR_ICC:

        /* TODO */

        break;

    case CSR_DCC:

        /* TODO */

        break;

    case CSR_EBA:

        tcg_gen_mov_tl(cpu_eba, cpu_R[dc->r1]);

        break;

    case CSR_DEBA:

        tcg_gen_mov_tl(cpu_deba, cpu_R[dc->r1]);

        break;

    case CSR_JTX:

        gen_helper_wcsr_jtx(cpu_env, cpu_R[dc->r1]);

        break;

    case CSR_JRX:

        gen_helper_wcsr_jrx(cpu_env, cpu_R[dc->r1]);

        break;

    case CSR_DC:

        gen_helper_wcsr_dc(cpu_env, cpu_R[dc->r1]);

        break;

    case CSR_BP0:

    case CSR_BP1:

    case CSR_BP2:

    case CSR_BP3:

        no = dc->csr - CSR_BP0;

        if (dc->num_breakpoints <= no) {

            qemu_log_mask(LOG_GUEST_ERROR,

                          "breakpoint #%i is not available\n", no);

            t_gen_illegal_insn(dc);

            break;

        }

        gen_helper_wcsr_bp(cpu_env, cpu_R[dc->r1], tcg_const_i32(no));

        break;

    case CSR_WP0:

    case CSR_WP1:

    case CSR_WP2:

    case CSR_WP3:

        no = dc->csr - CSR_WP0;

        if (dc->num_watchpoints <= no) {

            qemu_log_mask(LOG_GUEST_ERROR,

                          "watchpoint #%i is not available\n", no);

            t_gen_illegal_insn(dc);

            break;

        }

        gen_helper_wcsr_wp(cpu_env, cpu_R[dc->r1], tcg_const_i32(no));

        break;

    case CSR_CC:

    case CSR_CFG:

        qemu_log_mask(LOG_GUEST_ERROR, "invalid write access csr=%x\n",

                      dc->csr);

        break;

    default:

        qemu_log_mask(LOG_GUEST_ERROR, "write_csr: unknown csr=%x\n",

                      dc->csr);

        break;

    }

}
