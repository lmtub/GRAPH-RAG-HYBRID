static int qemu_rdma_exchange_get_response(RDMAContext *rdma,

                RDMAControlHeader *head, int expecting, int idx)

{

    uint32_t byte_len;

    int ret = qemu_rdma_block_for_wrid(rdma, RDMA_WRID_RECV_CONTROL + idx,

                                       &byte_len);



    if (ret < 0) {

        fprintf(stderr, "rdma migration: recv polling control error!\n");

        return ret;

    }



    network_to_control((void *) rdma->wr_data[idx].control);

    memcpy(head, rdma->wr_data[idx].control, sizeof(RDMAControlHeader));



    DDDPRINTF("CONTROL: %s receiving...\n", control_desc[expecting]);



    if (expecting == RDMA_CONTROL_NONE) {

        DDDPRINTF("Surprise: got %s (%d)\n",

                  control_desc[head->type], head->type);

    } else if (head->type != expecting || head->type == RDMA_CONTROL_ERROR) {

        fprintf(stderr, "Was expecting a %s (%d) control message"

                ", but got: %s (%d), length: %d\n",

                control_desc[expecting], expecting,

                control_desc[head->type], head->type, head->len);

        return -EIO;

    }

    if (head->len > RDMA_CONTROL_MAX_BUFFER - sizeof(*head)) {

        fprintf(stderr, "too long length: %d\n", head->len);

        return -EINVAL;

    }

    if (sizeof(*head) + head->len != byte_len) {

        fprintf(stderr, "Malformed length: %d byte_len %d\n",

                head->len, byte_len);

        return -EINVAL;

    }



    return 0;

}
