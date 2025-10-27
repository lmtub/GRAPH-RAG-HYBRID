static void vscsi_process_login(VSCSIState *s, vscsi_req *req)

{

    union viosrp_iu *iu = &req->iu;

    struct srp_login_rsp *rsp = &iu->srp.login_rsp;

    uint64_t tag = iu->srp.rsp.tag;



    trace_spapr_vscsi__process_login();



    /* TODO handle case that requested size is wrong and

     * buffer format is wrong

     */

    memset(iu, 0, sizeof(struct srp_login_rsp));

    rsp->opcode = SRP_LOGIN_RSP;

    /* Don't advertise quite as many request as we support to

     * keep room for management stuff etc...

     */

    rsp->req_lim_delta = cpu_to_be32(VSCSI_REQ_LIMIT-2);

    rsp->tag = tag;

    rsp->max_it_iu_len = cpu_to_be32(sizeof(union srp_iu));

    rsp->max_ti_iu_len = cpu_to_be32(sizeof(union srp_iu));

    /* direct and indirect */

    rsp->buf_fmt = cpu_to_be16(SRP_BUF_FORMAT_DIRECT | SRP_BUF_FORMAT_INDIRECT);



    vscsi_send_iu(s, req, sizeof(*rsp), VIOSRP_SRP_FORMAT);

}
