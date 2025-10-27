static uint16_t nvme_write_zeros(NvmeCtrl *n, NvmeNamespace *ns, NvmeCmd *cmd,

    NvmeRequest *req)

{

    NvmeRwCmd *rw = (NvmeRwCmd *)cmd;

    const uint8_t lba_index = NVME_ID_NS_FLBAS_INDEX(ns->id_ns.flbas);

    const uint8_t data_shift = ns->id_ns.lbaf[lba_index].ds;

    uint64_t slba = le64_to_cpu(rw->slba);

    uint32_t nlb  = le16_to_cpu(rw->nlb) + 1;

    uint64_t aio_slba = slba << (data_shift - BDRV_SECTOR_BITS);

    uint32_t aio_nlb = nlb << (data_shift - BDRV_SECTOR_BITS);



    if (slba + nlb > ns->id_ns.nsze) {

        return NVME_LBA_RANGE | NVME_DNR;

    }



    req->has_sg = false;

    block_acct_start(blk_get_stats(n->conf.blk), &req->acct, 0,

                     BLOCK_ACCT_WRITE);

    req->aiocb = blk_aio_pwrite_zeroes(n->conf.blk, aio_slba, aio_nlb,

                                        BDRV_REQ_MAY_UNMAP, nvme_rw_cb, req);

    return NVME_NO_COMPLETE;

}
