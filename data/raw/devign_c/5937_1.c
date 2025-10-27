static uint16_t nvme_dma_read_prp(NvmeCtrl *n, uint8_t *ptr, uint32_t len,

    uint64_t prp1, uint64_t prp2)

{

    QEMUSGList qsg;

    QEMUIOVector iov;

    uint16_t status = NVME_SUCCESS;



    if (nvme_map_prp(&qsg, &iov, prp1, prp2, len, n)) {

        return NVME_INVALID_FIELD | NVME_DNR;

    }

    if (qsg.nsg > 0) {

        if (dma_buf_read(ptr, len, &qsg)) {

            status = NVME_INVALID_FIELD | NVME_DNR;

        }

        qemu_sglist_destroy(&qsg);

    } else {

        if (qemu_iovec_to_buf(&iov, 0, ptr, len) != len) {

            status = NVME_INVALID_FIELD | NVME_DNR;

        }

        qemu_iovec_destroy(&iov);

    }

    return status;

}
