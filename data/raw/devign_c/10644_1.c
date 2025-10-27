static uint16_t nvme_set_feature(NvmeCtrl *n, NvmeCmd *cmd, NvmeRequest *req)

{

    uint32_t dw10 = le32_to_cpu(cmd->cdw10);

    uint32_t dw11 = le32_to_cpu(cmd->cdw11);



    switch (dw10) {

    case NVME_VOLATILE_WRITE_CACHE:

        blk_set_enable_write_cache(n->conf.blk, dw11 & 1);

        break;

    case NVME_NUMBER_OF_QUEUES:

        req->cqe.result =

            cpu_to_le32((n->num_queues - 1) | ((n->num_queues - 1) << 16));

        break;

    default:

        return NVME_INVALID_FIELD | NVME_DNR;

    }

    return NVME_SUCCESS;

}
