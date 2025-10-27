static size_t qemu_rdma_fill(RDMAContext *rdma, uint8_t *buf,

                             int size, int idx)

{

    size_t len = 0;



    if (rdma->wr_data[idx].control_len) {

        DDDPRINTF("RDMA %" PRId64 " of %d bytes already in buffer\n",

                    rdma->wr_data[idx].control_len, size);



        len = MIN(size, rdma->wr_data[idx].control_len);

        memcpy(buf, rdma->wr_data[idx].control_curr, len);

        rdma->wr_data[idx].control_curr += len;

        rdma->wr_data[idx].control_len -= len;

    }



    return len;

}
