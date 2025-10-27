static void dec_rcsr(DisasContext *dc)

{

    LOG_DIS("rcsr r%d, %d\n", dc->r2, dc->csr);



    switch (dc->csr) {

    case CSR_IE:

        tcg_gen_mov_tl(cpu_R[dc->r2], cpu_ie);

        break;

    case CSR_IM:

        gen_helper_rcsr_im(cpu_R[dc->r2], cpu_env);

        break;

    case CSR_IP:

        gen_helper_rcsr_ip(cpu_R[dc->r2], cpu_env);

        break;

    case CSR_CC:

        tcg_gen_mov_tl(cpu_R[dc->r2], cpu_cc);

        break;

    case CSR_CFG:

        tcg_gen_mov_tl(cpu_R[dc->r2], cpu_cfg);

        break;

    case CSR_EBA:

        tcg_gen_mov_tl(cpu_R[dc->r2], cpu_eba);

        break;

    case CSR_DC:

        tcg_gen_mov_tl(cpu_R[dc->r2], cpu_dc);

        break;

    case CSR_DEBA:

        tcg_gen_mov_tl(cpu_R[dc->r2], cpu_deba);

        break;

    case CSR_JTX:

        gen_helper_rcsr_jtx(cpu_R[dc->r2], cpu_env);

        break;

    case CSR_JRX:

        gen_helper_rcsr_jrx(cpu_R[dc->r2], cpu_env);

        break;

    case CSR_ICC:

    case CSR_DCC:

    case CSR_BP0:

    case CSR_BP1:

    case CSR_BP2:

    case CSR_BP3:

    case CSR_WP0:

    case CSR_WP1:

    case CSR_WP2:

    case CSR_WP3:

        cpu_abort(dc->env, "invalid read access csr=%x\n", dc->csr);

        break;

    default:

        cpu_abort(dc->env, "read_csr: unknown csr=%x\n", dc->csr);

        break;

    }

}
