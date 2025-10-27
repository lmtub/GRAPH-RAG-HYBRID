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

        tcg_gen_mov_tl(cpu_dc, cpu_R[dc->r1]);

        break;

    case CSR_BP0:

    case CSR_BP1:

    case CSR_BP2:

    case CSR_BP3:

        no = dc->csr - CSR_BP0;

        if (dc->env->num_bps <= no) {

            cpu_abort(dc->env, "breakpoint #%i is not available\n", no);

        }

        tcg_gen_mov_tl(cpu_bp[no], cpu_R[dc->r1]);

        break;

    case CSR_WP0:

    case CSR_WP1:

    case CSR_WP2:

    case CSR_WP3:

        no = dc->csr - CSR_WP0;

        if (dc->env->num_wps <= no) {

            cpu_abort(dc->env, "watchpoint #%i is not available\n", no);

        }

        tcg_gen_mov_tl(cpu_wp[no], cpu_R[dc->r1]);

        break;

    case CSR_CC:

    case CSR_CFG:

        cpu_abort(dc->env, "invalid write access csr=%x\n", dc->csr);

        break;

    default:

        cpu_abort(dc->env, "write_csr unknown csr=%x\n", dc->csr);

        break;

    }

}
