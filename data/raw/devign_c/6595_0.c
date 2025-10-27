static void nvme_clear_ctrl(NvmeCtrl *n)

{

    int i;



    for (i = 0; i < n->num_queues; i++) {

        if (n->sq[i] != NULL) {

            nvme_free_sq(n->sq[i], n);

        }

    }

    for (i = 0; i < n->num_queues; i++) {

        if (n->cq[i] != NULL) {

            nvme_free_cq(n->cq[i], n);

        }

    }



    bdrv_flush(n->conf.bs);

    n->bar.cc = 0;

}
