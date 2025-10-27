static uint16_t nvme_create_cq(NvmeCtrl *n, NvmeCmd *cmd)

{

    NvmeCQueue *cq;

    NvmeCreateCq *c = (NvmeCreateCq *)cmd;

    uint16_t cqid = le16_to_cpu(c->cqid);

    uint16_t vector = le16_to_cpu(c->irq_vector);

    uint16_t qsize = le16_to_cpu(c->qsize);

    uint16_t qflags = le16_to_cpu(c->cq_flags);

    uint64_t prp1 = le64_to_cpu(c->prp1);



    if (!cqid || !nvme_check_cqid(n, cqid)) {

        return NVME_INVALID_CQID | NVME_DNR;

    }

    if (!qsize || qsize > NVME_CAP_MQES(n->bar.cap)) {

        return NVME_MAX_QSIZE_EXCEEDED | NVME_DNR;

    }

    if (!prp1) {

        return NVME_INVALID_FIELD | NVME_DNR;

    }

    if (vector > n->num_queues) {

        return NVME_INVALID_IRQ_VECTOR | NVME_DNR;

    }

    if (!(NVME_CQ_FLAGS_PC(qflags))) {

        return NVME_INVALID_FIELD | NVME_DNR;

    }



    cq = g_malloc0(sizeof(*cq));

    nvme_init_cq(cq, n, prp1, cqid, vector, qsize + 1,

        NVME_CQ_FLAGS_IEN(qflags));

    return NVME_SUCCESS;

}
